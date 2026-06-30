"""Final decision resolver (Phase 19A, pure Python).

Single authority that merges the deterministic baseline with released LE 2.0
predictions through the Phase-18 ``ControlPolicy`` (the one confidence gate), then
the GuardLayer and DeviceAdapter, producing one ``DeviceControlCommand`` plus a
full ``DecisionTrace``. Both SHADOW and CONTROL flow through here; the only
difference is whether the released value is actually dispatched.

Priority (strictest first):
  1. safety / frost   2. user / mode lock   3. window / absence
  4. baseline         5. released LE2        6. device limits
  7. anti-chatter      8. single command
"""
from __future__ import annotations

from typing import Optional

from ..confidence import ConfidencePurpose
from ..models.boost import boost_offset_c_to_compat_factor
from ..runtime.control import (
    ControlContext, ControlFeature, ControlledPrediction, ControlPolicy)
from .contracts import (
    ControllerBaselineDecision, DecisionMode, DecisionTrace, DecisionTraceEntry,
    DeviceControlCommand, DispatchResult, FallbackReason, LearningPredictionSet,
    ZoneRuntimeInput)
from .device import DeviceAdapter
from .guards import GuardLayer

# feature -> (control feature, confidence purpose key)
_BOOST = (ControlFeature.BOOST_OFFSET, ConfidencePurpose.BOOST)
_PREHEAT = (ControlFeature.PREHEAT_START, ConfidencePurpose.PREHEAT)
_CUTOFF = (ControlFeature.EARLY_CUTOFF, ConfidencePurpose.EARLY_CUTOFF)


class FinalResolver:
    def __init__(self, *, policy: Optional[ControlPolicy] = None,
                 guards: Optional[GuardLayer] = None,
                 device: Optional[DeviceAdapter] = None,
                 boost_runtime_limit: Optional[float] = None) -> None:
        self.policy = policy or ControlPolicy(runtime_is_control=True)
        self.guards = guards or GuardLayer()
        self.device = device or DeviceAdapter()
        self.boost_runtime_limit = boost_runtime_limit

    # ------------------------------------------------------------------ helpers
    def _safety_locked(self, zin: ZoneRuntimeInput, base: ControllerBaselineDecision):
        if zin.frost_protection:
            return FallbackReason.SAFETY_OVERRIDE, "frost_protection"
        if zin.summer or base.summer:
            return FallbackReason.SAFETY_OVERRIDE, "summer"
        if zin.window_open or base.window_open:
            return FallbackReason.GUARD_BLOCKED, "window_open"
        if zin.absence:
            return FallbackReason.GUARD_BLOCKED, "absence"
        if zin.heating_failure:
            return FallbackReason.SAFETY_OVERRIDE, "heating_failure"
        return None, None

    def _control_context(self, zin: ZoneRuntimeInput) -> ControlContext:
        return ControlContext(
            window_open=zin.window_open, heating_failure=zin.heating_failure,
            frost_protection=zin.frost_protection,
            has_current_temperature=zin.indoor_temp_valid,
            boost_runtime_limit=self.boost_runtime_limit)

    # ------------------------------------------------------------------ resolve
    def resolve(self, zin: ZoneRuntimeInput, base: ControllerBaselineDecision,
                preds: LearningPredictionSet, *, mode: DecisionMode) -> tuple[DecisionTrace, DispatchResult]:
        entries: list[DecisionTraceEntry] = []
        reason_codes: set[str] = set()
        baseline_sp = base.trv_setpoint_c
        final_sp = baseline_sp
        applied_any = False

        fb, lock = self._safety_locked(zin, base)
        ctx = self._control_context(zin)

        def add(feature, base_v, le2_v, final_v, applied, reason, fb_reason, conf, clamp):
            entries.append(DecisionTraceEntry(
                feature=feature, baseline_value=base_v, le2_value=le2_v, final_value=final_v,
                applied=applied, reason=reason, fallback_reason=fb_reason,
                confidence=conf, clamp_applied=clamp))
            reason_codes.add(reason)

        # ---- boost-offset (full LE 2.0 authority for setpoint) ----
        # Correctness invariant: final_setpoint = tpi_baseline_setpoint + le2_boost_offset.
        # base.trv_setpoint_c is the TPI-computed baseline; dec.final_value is the additive
        # offset. Never: target_c + boost (that would discard the TPI baseline entirely).
        boost = preds.get("boost_offset")
        boost_ineligible_reason: Optional[str] = None
        if lock is None and boost is not None and base.trv_setpoint_c is not None:
            # Direct valve TRVs: heating authority is duty % alone; °C setpoint offset
            # does not apply. Boost is trace-only for these devices.
            if zin.tpi_valve_direct:
                boost_ineligible_reason = "direct_valve_control"
            elif base.target_c is None:
                # No comfort target available: cannot determine deficit → block boost.
                boost_ineligible_reason = "no_target"
            elif zin.boost_eligible is False:
                boost_ineligible_reason = "boost_not_eligible"
            elif zin.early_cutoff_state in ("cutoff_applied", "coasting_hold"):
                boost_ineligible_reason = "early_cutoff_hold"
            elif (zin.temperature_gap_c is not None
                  and zin.temperature_gap_c <= zin.boost_deficit_threshold_c):
                boost_ineligible_reason = "deficit_too_small"

        if lock is not None:
            add("boost_offset", base.boost_offset_c, boost.value if boost else None,
                base.boost_offset_c, False, lock, fb.value, None, 0.0)
        elif boost is None or base.trv_setpoint_c is None:
            add("boost_offset", base.boost_offset_c, None, base.boost_offset_c, False,
                "baseline", FallbackReason.NO_PREDICTION.value, None, 0.0)
        elif boost_ineligible_reason is not None:
            add("boost_offset", base.boost_offset_c, boost.value, base.boost_offset_c, False,
                boost_ineligible_reason, FallbackReason.GUARD_BLOCKED.value, None, 0.0)
        else:
            cr = preds.confidence_results.get(ConfidencePurpose.BOOST.value)
            cp = (ControlledPrediction(_BOOST[0], boost.value, boost.unit, cr,
                                       model_version=None, parameter_version=None)
                  if cr is not None else None)
            dec = self.policy.resolve(_BOOST[0], base.boost_offset_c, cp, context=ctx,
                                      unit=boost.unit)
            # Full LE 2.0 authority: apply when CONTROL mode and confidence gate passed.
            # Both increases and decreases are allowed (no reduce-only constraint).
            apply = (mode is DecisionMode.CONTROL and dec.applied
                     and dec.final_value is not None)
            if apply:
                # TPI authority: boost is ADDITIVE on top of the TPI baseline setpoint.
                # cand = tpi_baseline_setpoint + le2_boost_offset (never comfort_target + boost)
                cand = base.trv_setpoint_c + dec.final_value
                guard = self.guards.check_setpoint(baseline_sp, cand)
                if guard.allowed:
                    final_sp = guard.final_value
                    applied_any = True
                    for r in guard.reasons:
                        reason_codes.add(r)
                    add("boost_offset", base.boost_offset_c, boost.value, dec.final_value,
                        True, "le2_applied", FallbackReason.NONE.value,
                        dec.confidence.value if dec.confidence else None, dec.clamp_applied)
                else:
                    add("boost_offset", base.boost_offset_c, boost.value, base.boost_offset_c,
                        False, "anti_chatter" if "anti_chatter" in guard.reasons else "baseline",
                        FallbackReason.GUARD_BLOCKED.value, None, 0.0)
            else:
                add("boost_offset", base.boost_offset_c, boost.value,
                    base.boost_offset_c, False, "baseline",
                    (FallbackReason.SHADOW_MODE.value if mode is DecisionMode.SHADOW
                     else FallbackReason.CONTROL_REJECTED.value),
                    dec.confidence.value if dec.confidence else None, 0.0)

        # ---- advisory/trace-only features (never affect 19A setpoint) ----
        for key, (feat, purpose) in (("preheat_start", _PREHEAT), ("early_cutoff", _CUTOFF)):
            pred = preds.get(key)
            if pred is None:
                continue
            cr = preds.confidence_results.get(purpose.value)
            base_v = base.preheat_minutes if key == "preheat_start" else 0.0
            cp = (ControlledPrediction(feat, pred.value, pred.unit, cr,
                                       model_version=None, parameter_version=None)
                  if cr is not None else None)
            if lock is not None or cp is None:
                add(key, base_v, pred.value, base_v, False, lock or "baseline",
                    (fb.value if fb else FallbackReason.NO_PREDICTION.value), None, 0.0)
                continue
            dec = self.policy.resolve(feat, base_v, cp, context=ctx, unit=pred.unit)
            # 19A: integrated end-to-end but NOT applied to the immediate setpoint
            add(key, base_v, pred.value,
                dec.final_value if dec.applied else base_v, False,
                "le2_applied" if dec.applied else "baseline",
                FallbackReason.SHADOW_MODE.value if dec.applied else FallbackReason.CONTROL_REJECTED.value,
                dec.confidence.value if dec.confidence else None, dec.clamp_applied)

        command = self.device.build_command(
            zin.zone_id, setpoint_c=final_sp, duty_cycle=base.duty_cycle,
            source_reason="le2_applied" if applied_any else "baseline",
            shadow_only=(mode is DecisionMode.SHADOW or not applied_any))

        # Preheat time components: A = B + C (all three kept separate in trace).
        # A = preheat_command_lead_time_min (total lead time returned to coordinator)
        # B = effective_onset_delay_min (from ZoneRuntimeInput; adaptive when learned)
        # C = effective_room_heating_duration_min = A - B (only C drives HeatRate learning)
        _le2_total = zin.preheat_minutes_le2
        _onset_delay = zin.onset_delay_min or 5.0  # fallback to prior if not in input
        _room_heat_dur = round(_le2_total - _onset_delay, 2) if _le2_total is not None else None
        _ec_contribution = None
        _preheat_status = zin.preheat_status
        _preheat_fallback = _preheat_status in ("deterministic_baseline", "low_confidence")
        _selected = _le2_total  # selected == le2 value (baseline already folded in by shadow)

        # Boost trace fields: derive internal offset from final setpoint delta
        _boost_entry = next((e for e in entries if e.feature == "boost_offset"), None)
        _boost_offset_applied = (
            _boost_entry.final_value if _boost_entry and _boost_entry.applied else None)
        _boost_factor_compat = (
            boost_offset_c_to_compat_factor(_boost_offset_applied)
            if _boost_offset_applied is not None else None)
        # TPI authority chain: explicitly typed fields for no-double-apply audit
        _boost_rejection_reason = (
            _boost_entry.reason if _boost_entry and not _boost_entry.applied else None)
        _boost_requested_c = boost.value if boost is not None else None

        trace = DecisionTrace(
            zone_id=zin.zone_id, ts=zin.ts, mode=mode.value, decision_id=zin.decision_id,
            baseline_setpoint_c=baseline_sp, final_setpoint_c=command.trv_setpoint_c,
            applied_any=applied_any, entries=tuple(entries),
            reason_codes=tuple(sorted(reason_codes)),
            comfort_temperature_c=zin.comfort_temperature_c,
            early_cutoff_state=zin.early_cutoff_state,
            early_cutoff_hold_active=zin.early_cutoff_state in (
                "cutoff_applied", "coasting_hold"),
            preheat_minutes_le2=_le2_total,
            preheat_status=_preheat_status,
            preheat_baseline_minutes=zin.deterministic_baseline_preheat_min or base.preheat_minutes,
            selected_preheat_min=_selected,
            heat_rate_c_per_h=zin.heat_rate_c_per_h,
            heat_rate_confidence=zin.heat_rate_confidence,
            heat_rate_status=None,
            effective_onset_delay_min=_onset_delay,
            onset_delay_source=zin.onset_delay_status,
            onset_delay_status=zin.onset_delay_status,
            effective_room_heating_duration_min=_room_heat_dur,
            preheat_command_lead_time_min=_le2_total,
            temperature_gap_c=zin.temperature_gap_c,
            target_temperature_c=zin.target_temperature_c,
            context_time_bucket=zin.context_time_bucket,
            fallback_source=_preheat_status if _preheat_fallback else None,
            early_cutoff_contribution_c=_ec_contribution,
            preheat_fallback_used=_preheat_fallback,
            boost_offset_c_applied=_boost_offset_applied,
            boost_factor_compat=_boost_factor_compat,
            tpi_duty_percent=base.duty_cycle,
            tpi_baseline_setpoint_c=baseline_sp,
            boost_offset_requested_c=_boost_requested_c,
            boost_rejected_reason=_boost_rejection_reason,
        )
        dispatch = DispatchResult(
            zone_id=zin.zone_id, dispatched=applied_any and mode is DecisionMode.CONTROL,
            command=command, shadow_only=command.shadow_only,
            reason="le2_applied" if applied_any else "baseline")
        return trace, dispatch
