# Contributing to ThermoSmart

Thank you for your interest in ThermoSmart. Contributions of all kinds are welcome — bug reports, device testing, code improvements, documentation updates, and translations.

---

## Project Overview

ThermoSmart is a custom Home Assistant integration for self-learning, weather-aware TRV heating control. It learns thermal building behaviour over time and optimises heating decisions using a TPI controller, weather data, and presence awareness.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, release-ready code only |
| `develop` | Active development and testing |

- All pull requests must target `develop`, not `main`.
- Do not commit directly to `main`.
- `develop` does not maintain its own version series. Versions are bumped on `develop` only when preparing a release, then merged to `main`.

---

## Development Workflow

1. Fork the repository.
2. Create a branch from `develop`:
   ```
   git checkout -b fix/your-description develop
   git checkout -b feature/your-description develop
   ```
3. Implement your changes.
4. Test locally (see below).
5. Open a pull request against `develop`.

Direct pull requests against `main` are only accepted if explicitly agreed with the maintainer in advance.

---

## Local Testing

The CI workflow runs HACS validation and hassfest on every push and pull request.

For feature or integration testing, use a live Home Assistant instance:

1. Copy the contents of `custom_components/thermosmart/` into your Home Assistant `custom_components/` directory.
2. Restart Home Assistant fully.
3. Check the HA logs for errors or unexpected behaviour.
4. Verify the affected feature or fix in your specific setup.

**Do not commit** private configuration files, log files, `.storage` entries, or any file containing personal data.

---

## Pull Request Requirements

Every pull request should include:

- A clear description of the change
- The reason for the change
- Which devices or platforms are affected
- What was tested and how
- Any known side effects or regressions
- Relevant log excerpts or screenshots, if applicable

---

## Device Compatibility

### Status Categories

All device or platform compatibility claims must use one of these categories:

| Category | Meaning |
|----------|---------|
| **Developer-tested** | Confirmed working by the project maintainer in a live setup |
| **Community-tested** | Confirmed working by a community member — must be documented (see below) |
| **Code-expected** | Auto-detection patterns exist; behaviour is expected but unconfirmed |
| **Experimental** | Partial support or uncertain behaviour; use with caution |
| **Not tested** | No test data available |
| **Not recommended** | Known conflicts or architectural incompatibilities |

**A detected capability is not a compatibility guarantee.**
**A setpoint fallback does not constitute full support.**

A device may only be listed as *Developer-tested* or *Community-tested* after a real-world test in a live Home Assistant setup has been completed and documented.

### Documenting Community-Tested Devices

When reporting a successfully tested device, include:

- Manufacturer and model name
- Firmware version
- Integration used (ZHA, Zigbee2MQTT, Z-Wave JS, Homematic, or other)
- Available entities (valve, calibration, temperature input, etc.)
- ThermoSmart control mode used (Direct Valve Control or Setpoint Boost)
- Behaviour of Direct Valve Control, if applicable
- Behaviour of external temperature input, if applicable
- Calibration entity used, if applicable
- Test result and any known limitations

---

## Releases

- Releases are created exclusively from `main` by the maintainer.
- GitHub releases and tags are not created by contributors.
- Pre-releases are used for significant or higher-risk changes during the beta phase.
- Do not change the version number in `manifest.json` or `const.py` unless explicitly requested by the maintainer.

---

## ThermoSmart Card

The Lovelace card is maintained in a separate repository. Card-related changes belong there, not in this repository.

If your pull request requires coordinated changes to both the integration and the card, state this clearly in the PR description and reference the corresponding card PR.

---

## Security and Privacy

- Do not commit passwords, tokens, API keys, private IP addresses, or personal configuration.
- Review log excerpts before posting them — Home Assistant logs can contain entity IDs, device names, and network details.
- Do not publish full Home Assistant diagnostic dumps without reviewing them for sensitive data first.

If you discover a security vulnerability, please report it privately to the maintainer rather than opening a public issue.

---

## License

ThermoSmart is licensed under the [MIT License](LICENSE).

By submitting a contribution, you agree that your changes will be published under the same license.

- Do not include code from projects with incompatible licenses.
- Any third-party code or derived content must be clearly attributed and confirmed to be license-compatible.
