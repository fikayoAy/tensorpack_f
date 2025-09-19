# TensorPack License Management Specification

## Table of Contents
- [License Storage](#license-storage)
- [Activation Process](#activation-process)
- [License Types](#license-types)
- [Verification Methods](#verification-methods)
- [Machine Binding](#machine-binding)
- [Edge Cases and Caveats](#edge-cases-and-caveats)
- [API Reference](#api-reference)

## License Storage

- Local cache: license state is persisted under `~/.tensorpack/license/license_data.json` and usage under `~/.tensorpack/license/usage_data.json`. Local per-key license files are stored as `~/.tensorpack/license/<license_key>.json`.

## Activation Process

- Activation flow: `LicenseManager.activate_license()` tries (in order):
  1. Offline signature verification of a local signed license file (if `cryptography` is available).
  2. Online verification against Cryptolens (HTTP API).
  3. If the key corresponds to a local signed file, it can accept that as a fallback when online checks fail.
- What activation does: on successful verification the manager updates `self._license_data` (license key, type, features, expiry, `activated` flag) and calls `_save_license_data()` — which overwrites `license_data.json`.
- Trial generation: `generate_license()` can create local trial/academic license files; trials are recorded with `license_type='trial'` and an `expires` timestamp.
- Verification: `verify_license()` prefers online Cryptolens check, then offline verification, and finally the cached activation data (`activated` flag) as a last fallback.

## Machine Binding

Free licenses are now bound to the machine they are generated on to prevent abuse. The machine ID is calculated using hardware identifiers and stored with the license. During verification, the current machine ID is compared to the stored one.

## Edge Cases and Caveats

- Server-side constraints: If the licensing server (Cryptolens) enforces activation/usage limits per key or per-machine, the server may refuse activation. In that case activation will fail and the trial remains in place.
- Offline signed-license verification requires the `cryptography` package; without it the manager relies on online Cryptolens or a plain local license file.
- If a user has multiple cached local license files, `activate_license()` prefers verifying the key via Cryptolens first; a local `<key>.json` can be used if online verification fails.
- Usage counters (`usage_data.json`) are separate and are not necessarily reset on activation; activation updates features but may leave usage history intact.

## Conclusion

When a new paid license key is successfully activated, it replaces/overwrites the prior cached license state locally. The repository stores the new license details into the same `license_data.json`, so a paid license will supersede a trial on the machine where activation succeeds.

Practical step-by-step upgrade & verification (copyable)

1) Inspect existing license files and cached state

```powershell
Get-ChildItem $env:USERPROFILE\\.tensorpack\\license\\* -ErrorAction SilentlyContinue
Get-Content $env:USERPROFILE\\.tensorpack\\license\\license_data.json -Raw | ConvertFrom-Json
```

2) Activate the new license key (preferred)

- If the `tensorpack` console script works:

```powershell
tensorpack --activate-license YOUR-KEY
```

- If `tensorpack` console script is not available, run the module or call the manager from Python:

```powershell
# via module entry (recommended when installed)
python -m tensorpack --activate-license YOUR-KEY

# direct Python call (works in development without installing console script)
python -c "from tensorpack.license_manager import LicenseManager; lm=LicenseManager(); print(lm.activate_license('YOUR-KEY'))"
```

3) Verify license status after activation

```powershell
tensorpack --license-info
# or
python -m tensorpack --license-info
```

4) Offline fallback using a signed license file

If the vendor provided a signed license file, place it at:

```
%USERPROFILE%\\.tensorpack\\license\\<LICENSE_KEY>.json
```

Then run the same `--activate-license <LICENSE_KEY>` command; the manager will attempt offline verification using that file.

5) If activation fails (common causes and fixes)

- Server rejects activation (activation limits, revoked key): contact vendor support with your new key and machine/account details so they can revoke previous trial activations or adjust limits.
- Missing `cryptography` for offline verification: install it in your venv:

```powershell
pip install cryptography requests
```

- `tensorpack` console script fails locally: install the package into your venv from the repo root:

```powershell
python -m pip install -e .
```

What changes on disk when activation succeeds

- `~/.tensorpack/license/license_data.json` will be updated with fields similar to:

```json
{
  "license_key": "YOUR-KEY",
  "license_type": "premium",
  "activated": true,
  "expires": "2026-09-01T12:00:00Z",
  "features": { /* premium entitlements */ },
  "activation_date": "2025-09-13T...Z"
}
```

When to contact the vendor / escalation

- Activation rejected due to server-side limits or revocation: contact vendor support with the new key, purchase/order details, and machine identifier so they can resolve activations or issue a replacement.
- If offline activation fails and you have a signed license file, provide that file to support and ask for instructions or a reissued signed license.


