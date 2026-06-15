# Security Policy

## Supported Versions

Security fixes are applied to the latest stable release only.

| Version | Supported |
|---------|-----------|
| Latest stable (`v1.1.x`) | ✅ Yes |
| Older versions | ❌ No — please update |

---

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub Issues.**

Please use [GitHub Security Advisories](https://github.com/Mikasmarthome/ThermoSmart/security/advisories/new) to report vulnerabilities privately. This allows the maintainer to assess and address the issue before any public disclosure.

When reporting, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- The affected version(s)
- Any potential impact or attack scenario you have identified
- If applicable: suggested remediation or patch

You will receive an acknowledgement as soon as possible. The maintainer aims to respond to security reports within **7 days**.

---

## Responsible Disclosure

ThermoSmart follows a coordinated disclosure approach:

1. The reporter submits a private Security Advisory on GitHub.
2. The maintainer acknowledges receipt and investigates.
3. A fix is developed and tested on `develop`.
4. A patched release is published.
5. The Security Advisory is published after the fix is available.

Please allow reasonable time for a fix before any public disclosure. If no response is received within **14 days**, reporters are encouraged to follow up before any public disclosure.

---

## Security Response Process

| Step | Target Timeframe |
|------|-----------------|
| Acknowledgement of report | 7 days |
| Initial assessment | 14 days |
| Fix released (critical) | As soon as possible |
| Fix released (non-critical) | Next regular release |
| Advisory published | After fix is available |

---

## Scope

### ThermoSmart Integration

Repository: [Mikasmarthome/ThermoSmart](https://github.com/Mikasmarthome/ThermoSmart)

In scope:

- Code execution or privilege escalation via the integration
- Unintended exposure of Home Assistant credentials or tokens
- Manipulation of heating control in a way that could cause physical harm (e.g. disabling frost protection)
- Storage or transmission of user data beyond the local Home Assistant instance

Out of scope:

- Vulnerabilities in Home Assistant core itself
- Issues in third-party TRV firmware or hardware
- Denial-of-service through excessive polling of the local HA instance
- Issues that require physical access to the device running Home Assistant

### ThermoSmart Card

Repository: [Mikasmarthome/thermosmart-card](https://github.com/Mikasmarthome/thermosmart-card)

The card is a Lovelace frontend component with no backend logic. Security reports for the card follow the same process via the card repository's Security Advisories.

---

## General Notes

ThermoSmart runs entirely **locally** within your Home Assistant instance. It does not communicate with any external server, does not collect telemetry, and does not transmit any data outside your local network.

Normal bug reports (non-security) should be filed as [GitHub Issues](https://github.com/Mikasmarthome/ThermoSmart/issues).
