"""
Simplified LicenseManager using Cryptolens for license verification and
PayPal payment links for purchases. This file replaces the previous
cryptography-based signature verification and PayPal API flow.

Notes:
- Cryptolens: Product ID and API token are configured via constants below.
  The token should be kept secret on the vendor side; the client only needs
  product-based verification using Cryptolens public endpoints.
- Payment: a PayPal payment link is provided for customers; no PayPal API
  integration is required.

This implementation keeps the original feature access logic and offline
fallbacks (cached license_data). It attempts a Cryptolens server call to
validate a license key when available, and otherwise falls back to locally
stored license info.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import json
import datetime
import os
import webbrowser
import requests
import base64
import xml.etree.ElementTree as ET

# Try to import cryptography for RSA signature verification (public key only)
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False
    logger = logging.getLogger('tensorpack.license')
    logger.warning('cryptography not installed — offline signature verification will be unavailable. Install with: pip install cryptography')

CRYPTOLENS_PRODUCT_ID = 30951  # Must be int or string of int, not 'Id30951'
CRYPTOLENS_API_TOKEN = "WyIxMTIyMDU1NDUiLCJNU0lJZWhjaWhrZHJkMEJ0Q3ZjU0plLzlhRGl6dXJLTTNKZXpFM3lCIl0="
# Use the correct Cryptolens endpoint for key verification
CRYPTOLENS_VERIFY_URL = "https://app.cryptolens.io/api/key/GetKey"

# PayPal payment link (user requested specific link for purchases)
PAYPAL_PAYMENT_LINK = "https://www.paypal.com/ncp/payment/8W7H55GYX66X8"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tensorpack.license')


class LicenseManager:
    """License management using Cryptolens for verification.

    Behaviour:
    - Attempts online verification with Cryptolens when a license key is present.
    - If online verification fails or network is unavailable, falls back to cached
      license data stored under ~/.tensorpack/license/license_data.json.
    - Uses simple local generation for trial/academic licenses (vendor should
      prefer issuing licenses via Cryptolens for paid licenses).
    """

    def __init__(self):
        # Product identifier used in payloads and messages
        self.PRODUCT_ID = "tensorpack-premium"

        # Cryptolens product configuration
        self.cryptolens_product = CRYPTOLENS_PRODUCT_ID
        self.cryptolens_token = CRYPTOLENS_API_TOKEN
        self.cryptolens_url = CRYPTOLENS_VERIFY_URL

        # Academic domains that get premium features automatically
        self.academic_domains = [".ac.uk"]

        # Pricing and trial settings
        self.STANDARD_PRICE = 850.00
        self.TRIAL_DAYS = 7

        # Upgrade URL and PayPal link
        self.UPGRADE_URL = "https://fikayoAy.github.io/tensorpack/"
        self.PAYPAL_LINK = PAYPAL_PAYMENT_LINK

        # Feature access configuration (unchanged)
        self.feature_access = {
            "free": {
                "tensor_to_matrix": True,
                "matrix_to_tensor": True,
                "normalization": True,
                "traverse_graph": True,
                "discover_connections": True,
                
                "custom_transformations": False,
                "list_transforms": True,
                "describe_transforms": True,
                "remove_transforms": False,
                "export_formats": ["json"],
                "visualization": True,
                "advanced_visualizations": False,
                "max_datasets": 3,
                "max_file_size_mb": 500,
                "concurrent_operations": 1,
                "daily_api_calls": 5,
                "daily_advanced_operations": 1
            },
            "free_trial": {
                "tensor_to_matrix": True,
                "matrix_to_tensor": True,
                "normalization": True,
                "traverse_graph": True,
                "discover_connections": True,
               
                "custom_transformations": True,
                "list_transforms": True,
                "describe_transforms": True,
                "remove_transforms": True,
                "export_formats": ["json", "csv", "xlsx", "parquet", "html", "md", "sqlite"],
                "visualization": True,
                "advanced_visualizations": True,
                "max_datasets": float('inf'),
                "max_file_size_mb": float('inf'),
                "concurrent_operations": float('inf')
            },
            "premium": {
                "tensor_to_matrix": True,
                "matrix_to_tensor": True,
                "normalization": True,
                "traverse_graph": True,
                "discover_connections": True,
               
                "custom_transformations": True,
                "list_transforms": True,
                "describe_transforms": True,
                "remove_transforms": True,
                "export_formats": ["json", "csv", "xlsx", "parquet", "html", "md", "sqlite"],
                "visualization": True,
                "advanced_visualizations": True,
                "max_datasets": float('inf'),
                "max_file_size_mb": float('inf'),
                "concurrent_operations": float('inf')
            }
        }

        # Path to store license and usage data
        self.license_dir = Path.home() / '.tensorpack' / 'license'
        self.license_dir.mkdir(parents=True, exist_ok=True)
        self.license_file = self.license_dir / 'license_data.json'
        self.usage_file = self.license_dir / 'usage_data.json'

        # Load cached data
        self._license_data = self._load_license_data()
        self._usage_data = self._load_usage_data()

        # Advanced features list
        self.advanced_features = ["traverse_graph", "discover_connections", "combine_matrices"]

        # Embedded RSA public key (Cryptolens) supplied by user as RSAKeyValue XML
        # This will be used for offline signature verification of license blobs.
        # Replace or keep as provided by the user.
        self._embedded_rsa_xml = (
            "<RSAKeyValue><Modulus>xLEp4B5IPxa7gCN6jnBpQs3c84FIZftzfAisdl13tC2Um0t7DBJjPr8uEA5076tJwMoVllfh4drfYVBhQsmE0+uv5btUm3j9gPXcrrAeUFMyW+UZs/yqEqHF8yOOJk4j+V8WTGDYoEe196HjVVpLiCmelLfd1odOQfjvtEUkgeL3DSE6R7i+w2JnDMEvENRtMX249sBr6dOQHj8Zj99wldsMUDAQ76xTkxemrcdMgfFoK3xdwsW4bjs49xpFui0sc0u7snPe1AqxGaHqTkbMh6vSEnELJVVyjpTQhIr/66C9Lf/omY049WMjteKQmbD+/CRUJ6g+uHseg1nuraoiew==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
        )

        # Cache for parsed public key
        self._embedded_public_key = None

    # ----------------------- Persistence helpers ----------------------------
    def _load_license_data(self) -> Dict[str, Any]:
        if self.license_file.exists():
            try:
                with open(self.license_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load license file: {e}")
        # Default structure
        return {
            'license_key': None,
            'customer_id': None,
            'license_type': 'free',
            'activated': False,
            'expires': None,
            'activation_date': None,
            'machine_code': None,
            'features': self.feature_access['free'],
            'metadata': {}
        }

    def _save_license_data(self) -> bool:
        try:
            with open(self.license_file, 'w', encoding='utf-8') as f:
                json.dump(self._license_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving license data: {e}")
            return False

    def _load_usage_data(self) -> Dict[str, Any]:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r', encoding='utf-8') as f:
                    usage = json.load(f)
                    if usage.get('date') != today:
                        usage['date'] = today
                        usage['api_calls'] = 0
                        usage['advanced_operations'] = 0
                    return usage
            except Exception:
                pass
        return {'date': today, 'api_calls': 0, 'advanced_operations': 0, 'feature_usage': {}}

    def _save_usage_data(self) -> bool:
        try:
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(self._usage_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")
            return False

    # ----------------------- Cryptolens integration -------------------------
    def _verify_with_cryptolens(self, license_key: str) -> Tuple[bool, Dict[str, Any]]:
        """Verify license key against Cryptolens and return (valid, data).

        Uses the /v3/License/Key endpoint. Returns (False, {}) on network/error.
        """
        try:
            params = {
                'ProductId': str(self.cryptolens_product),  # Must be string or int, not 'Id30951'
                'Key': license_key,
                'token': self.cryptolens_token
            }
            resp = requests.get(self.cryptolens_url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            # Cryptolens returns result==0 for success, result==1 for not found, etc.
            if data.get('result') == 0 and 'licenseKey' in data:
                return True, data['licenseKey']
            # Fallback: if 'License' key exists (older API)
            if data.get('License'):
                return True, data['License']
            # Some responses include 'Success' flag
            if data.get('Success'):
                return True, data
            return False, data
        except Exception as e:
            logger.debug(f"Cryptolens verification failed: {e}")
            return False, {}

    # ----------------------- Offline signature verification -----------------
    def _parse_rsa_xml_to_public_key(self, rsa_xml: str):
        """Parse RSAKeyValue XML (Modulus/Exponent base64) to a cryptography public key."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError('cryptography not available')

        try:
            root = ET.fromstring(rsa_xml)
            mod_b64 = root.findtext('Modulus')
            exp_b64 = root.findtext('Exponent')
            if not mod_b64 or not exp_b64:
                raise ValueError('Invalid RSAKeyValue XML')
            n = int.from_bytes(base64.b64decode(mod_b64), 'big')
            e = int.from_bytes(base64.b64decode(exp_b64), 'big')
            pub_numbers = RSAPublicNumbers(e, n)
            return pub_numbers.public_key(default_backend())
        except Exception as e:
            raise

    def verify_offline(self, license_key_or_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt offline verification of a signed license file.

        Expects a license JSON at ~/.tensorpack/license/<license_key>.json or a full path.
        The license must contain at minimum: {"data": {...}, "signature": "hex..."}

        Returns (True, data) on success, or (False, {}) on failure.
        """
        if not CRYPTO_AVAILABLE:
            logger.warning('Offline verification requested but cryptography not installed')
            return False, {}

        # Resolve path
        p = Path(license_key_or_path)
        if not p.exists():
            # try license_dir lookup
            p = self.license_dir / f"{license_key_or_path}.json"
            if not p.exists():
                return False, {}

        try:
            with open(p, 'r', encoding='utf-8') as f:
                full = json.load(f)
        except Exception:
            return False, {}

        # Accept both legacy containing data+signature or simple license data
        data = full.get('data') or full
        signature_hex = full.get('signature')
        # If signature not present, offline verification cannot proceed
        if not signature_hex:
            return False, {}

        try:
            sig = bytes.fromhex(signature_hex)
        except Exception:
            return False, {}

        # Prepare canonical bytes for verification
        try:
            license_bytes = json.dumps(data, sort_keys=True).encode()
        except Exception:
            return False, {}

        # Load embedded public key if not cached
        try:
            if self._embedded_public_key is None:
                self._embedded_public_key = self._parse_rsa_xml_to_public_key(self._embedded_rsa_xml)

            pub = self._embedded_public_key
            pub.verify(
                sig,
                license_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            # signature valid — extract features/tier/expiry
            return True, data
        except Exception as e:
            logger.debug(f"Offline signature verification failed: {e}")
            return False, {}

    # ----------------------- Email helpers --------------------------------
    def _is_academic_email(self, email: str) -> bool:
        if not email:
            return False
        email = email.lower()
        for domain in self.academic_domains:
            if email.endswith(domain):
                logger.info(f"Academic email detected: {email}")
                return True
        return False

    # ----------------------- License generation ----------------------------
    def generate_license(self, email: str, license_type: str = 'premium') -> Tuple[bool, str]:
        """Create a local license file for trials/academic users. For paid licenses,
        vendor should create a license in Cryptolens and provide the key to the user.
        """
        if not email:
            return False, "Email required"

        expires = None
        if license_type == 'trial':
            expires = (datetime.datetime.now() + datetime.timedelta(days=self.TRIAL_DAYS)).isoformat()

        license_key = f"TP-{email.split('@')[0]}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        license_data = {
            'email': email,
            'license_key': license_key,
            'license_type': license_type,
            'product': self.PRODUCT_ID,
            'issued_at': datetime.datetime.now().isoformat(),
            'expires': expires,
            'status': 'active'
        }

        # Academic override
        if self._is_academic_email(email):
            license_data['license_type'] = 'academic'

        try:
            with open(self.license_dir / f"{license_key}.json", 'w', encoding='utf-8') as f:
                json.dump(license_data, f, indent=2)
            logger.info(f"Local license generated: {license_key}")
            return True, license_key
        except Exception as e:
            logger.error(f"Failed to write license file: {e}")
            return False, str(e)

    # ----------------------- Activation & processing ----------------------
    def activate_license(self, license_key_or_order: str) -> Tuple[bool, str]:
        """Activate using either a license key (preferred) or a special order token.

        The previous PayPal API flow is removed; users should pay using the provided
        PayPal link and the vendor will issue a Cryptolens license key which the
        customer pastes into the product via --activate-license.
        """
        # If the input looks like a PayPal order reference, instruct user to use the link
        if license_key_or_order.startswith('PAYPAL-') or license_key_or_order.lower().startswith('order-'):
            return False, f"Please complete payment via: {self.PAYPAL_LINK} and then paste your license key here."

        license_key = license_key_or_order
        license_path = self.license_dir / f"{license_key}.json"

        # Offline-first: try embedded RSA signature verification if available
        if CRYPTO_AVAILABLE:
            off_ok, off_data = self.verify_offline(license_key)
            if off_ok:
                email = off_data.get('email')
                license_type = off_data.get('license_type', 'premium')
                expires = off_data.get('expires')
                is_academic = self._is_academic_email(email) if email else False
                features = self.feature_access['premium'] if (is_academic or license_type == 'premium') else (
                    self.feature_access['free_trial'] if license_type == 'trial' else self.feature_access['free']
                )
                self._license_data = {
                    'license_key': license_key,
                    'customer_id': email,
                    'license_type': 'academic' if is_academic else license_type,
                    'activated': True,
                    'activation_date': datetime.datetime.now().isoformat(),
                    'expires': expires,
                    'machine_code': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME'),
                    'features': features,
                    'is_academic': is_academic,
                    'metadata': {'created': off_data.get('issued_at') or off_data.get('Created')}
                }
                self._save_license_data()
                logger.info(f"License activated offline: {license_key}")

                # Attempt an online verification to refresh/revoke status if network available
                try:
                    ok_online, data_online = self._verify_with_cryptolens(license_key)
                    if ok_online:
                        email = data_online.get('Email') or data_online.get('email') or email
                        expires = data_online.get('ExpirationDate') or data_online.get('expires') or expires
                        self._license_data.update({
                            'customer_id': email,
                            'expires': expires,
                        })
                        self._save_license_data()
                except Exception:
                    pass

                return True, "License activated (offline)."

        # First try to verify with Cryptolens (if network available)
        ok, data = self._verify_with_cryptolens(license_key)
        if ok:
            # Map Cryptolens response to internal fields
            email = data.get('Email') or data.get('email') or self._license_data.get('customer_id')
            license_type = 'premium'
            expires = data.get('ExpirationDate') or data.get('expires')

            features = self.feature_access['premium']
            self._license_data = {
                'license_key': license_key,
                'customer_id': email,
                'license_type': license_type,
                'activated': True,
                'activation_date': datetime.datetime.now().isoformat(),
                'expires': expires,
                'machine_code': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME'),
                'features': features,
                'is_academic': self._is_academic_email(email) if email else False,
                'metadata': {'created': data.get('Created')}
            }
            self._save_license_data()
            logger.info(f"License activated via Cryptolens: {license_key}")
            return True, "License activated successfully. Thank you!"

        # If Cryptolens check failed, require offline-signed license file to be valid
        if license_path.exists():
            try:
                # Development override: if env var TENSORPACK_DEV_ACTIVATE==1, accept unsigned local license JSON
                if os.environ.get('TENSORPACK_DEV_ACTIVATE') == '1':
                    try:
                        # Accept files that may include a UTF-8 BOM by using utf-8-sig
                        with open(license_path, 'r', encoding='utf-8-sig') as f:
                            off_data = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to read local license file for dev-activation: {e}")
                        return False, f"Failed to read local license file: {e}"

                    email = off_data.get('email')
                    license_type = off_data.get('license_type', 'premium')
                    expires = off_data.get('expires')
                    is_academic = self._is_academic_email(email) if email else False
                    features = self.feature_access['premium'] if (is_academic or license_type == 'premium') else (
                        self.feature_access['free_trial'] if license_type == 'trial' else self.feature_access['free']
                    )
                    self._license_data = {
                        'license_key': license_key,
                        'customer_id': email,
                        'license_type': 'academic' if is_academic else license_type,
                        'activated': True,
                        'activation_date': datetime.datetime.now().isoformat(),
                        'expires': expires,
                        'machine_code': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME'),
                        'features': features,
                        'is_academic': is_academic,
                        'metadata': {'created': off_data.get('issued_at') or off_data.get('Created')}
                    }
                    self._save_license_data()
                    logger.info(f"Local license activated (dev override): {license_key}")
                    return True, "Local license activated (dev override)."

                # Only accept local license files if offline signature verification succeeds
                off_ok, off_data = self.verify_offline(str(license_path))
                if not off_ok:
                    logger.warning(f"Local license file present but failed offline verification: {license_path}")
                    return False, "Local license present but signature verification failed. Please activate online or provide a valid signed license."

                # Use the verified data
                email = off_data.get('email')
                license_type = off_data.get('license_type', 'premium')
                expires = off_data.get('expires')
                is_academic = self._is_academic_email(email) if email else False
                features = self.feature_access['premium'] if (is_academic or license_type == 'premium') else (
                    self.feature_access['free_trial'] if license_type == 'trial' else self.feature_access['free']
                )
                self._license_data = {
                    'license_key': license_key,
                    'customer_id': email,
                    'license_type': 'academic' if is_academic else license_type,
                    'activated': True,
                    'activation_date': datetime.datetime.now().isoformat(),
                    'expires': expires,
                    'machine_code': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME'),
                    'features': features,
                    'is_academic': is_academic,
                    'metadata': {'created': off_data.get('issued_at') or off_data.get('Created')}
                }
                self._save_license_data()
                logger.info(f"Local license activated (verified): {license_key}")
                return True, "Local signed license activated."
            except Exception as e:
                logger.error(f"Failed to activate local license: {e}")
                return False, f"Failed to activate license: {e}"

        return False, f"License verification failed. Please purchase via: {self.PAYPAL_LINK}"

    # ----------------------- Verification ---------------------------------
    def verify_license(self) -> Tuple[bool, str]:
        """Verify currently stored license; prefer Cryptolens online check, else cached data."""
        if self._license_data.get('license_type') == 'free':
            return True, 'Free tier license is valid.'

        license_key = self._license_data.get('license_key')
        if not license_key:
            return False, 'No license found. Please activate a license.'

        # Offline-first verification
        if CRYPTO_AVAILABLE:
            off_ok, off_data = self.verify_offline(license_key)
            if off_ok:
                email = off_data.get('email')
                license_type = off_data.get('license_type', 'premium')
                expires = off_data.get('expires')
                is_academic = self._is_academic_email(email) if email else False
                features = self.feature_access['premium'] if (is_academic or license_type == 'premium') else self.feature_access['free']
                self._license_data.update({
                    'license_type': 'academic' if is_academic else license_type,
                    'features': features,
                    'expires': expires,
                    'is_academic': is_academic,
                })
                self._save_license_data()
                logger.info('License verified offline via embedded RSA key')

                # Try online verification to refresh data if network available
                try:
                    ok_on, data_on = self._verify_with_cryptolens(license_key)
                    if ok_on:
                        email = data_on.get('Email') or data_on.get('email') or email
                        expires = data_on.get('ExpirationDate') or data_on.get('expires') or expires
                        self._license_data.update({'customer_id': email, 'expires': expires})
                        self._save_license_data()
                        logger.info('Refreshed license info from Cryptolens')
                except Exception:
                    pass

                return True, f"License verified offline: {self._license_data.get('license_type')}"

        # Fallback: try Cryptolens online verification
        ok, data = self._verify_with_cryptolens(license_key)
        if ok:
            # Update internal cache
            email = data.get('Email') or data.get('email')
            license_type = 'premium'
            expires = data.get('ExpirationDate') or data.get('expires')
            is_academic = self._is_academic_email(email) if email else False
            features = self.feature_access['premium'] if (is_academic or license_type == 'premium') else self.feature_access['free']
            self._license_data.update({
                'license_type': 'academic' if is_academic else license_type,
                'features': features,
                'expires': expires,
                'is_academic': is_academic,
            })
            self._save_license_data()
            logger.info('License verified via Cryptolens')
            return True, 'License verified: Premium'

        # Fall back to cached/local license file verification only if it verifies offline
        try:
            license_path = self.license_dir / f"{license_key}.json"
            if license_path.exists():
                off_ok, off_data = self.verify_offline(str(license_path))
                if not off_ok:
                    logger.debug('Local license present but offline verification failed; will not trust unsigned local files.')
                else:
                    email = off_data.get('email')
                    license_type = off_data.get('license_type', 'premium')
                    expires = off_data.get('expires')
                    is_academic = self._is_academic_email(email) if email else False
                    if expires and license_type == 'trial' and not is_academic:
                        try:
                            expires_date = datetime.datetime.fromisoformat(expires)
                            if expires_date < datetime.datetime.now():
                                self._license_data['license_type'] = 'free'
                                self._license_data['features'] = self.feature_access['free']
                                self._save_license_data()
                                return True, 'Trial expired. Downgraded to free.'
                        except ValueError:
                            pass
                    # Update cache from verified data
                    self._license_data.update({
                        'license_type': 'academic' if is_academic else license_type,
                        'features': self.feature_access['premium'] if is_academic or license_type == 'premium' else self.feature_access['free_trial'],
                        'expires': expires,
                        'is_academic': is_academic
                    })
                    self._save_license_data()
                    return True, f"License valid ({self._license_data['license_type']}). Offline mode."
        except Exception as e:
            logger.debug(f"Local verification failed: {e}")

        # Final fallback: use cached activation if available
        if self._license_data.get('activated'):
            return True, f"License valid ({self._license_data.get('license_type')}). Offline mode."

        return False, 'License verification failed.'

    # ----------------------- Feature helpers -------------------------------
    def check_feature_access(self, feature_name: str) -> Tuple[bool, str]:
        # Normalize common aliases (plural/synonym) to the canonical feature keys
        alias_map = {
            'visualizations': 'visualization',
            'visualisation': 'visualization',
            'visuals': 'visualization',
            'advanced-visualizations': 'advanced_visualizations',
            'advanced_visualisation': 'advanced_visualizations'
        }
        feature_name = alias_map.get(feature_name, feature_name)

        is_valid, message = self.verify_license()
        if not is_valid:
            return False, message

        features = self._license_data.get('features', {})
        has_access = features.get(feature_name, False)
        if not has_access:
            license_type = self._license_data.get('license_type', 'none')
            if license_type == 'trial':
                return False, f"Feature '{feature_name}' requires a premium license. Your trial does not include this feature."
            else:
                return False, f"Feature '{feature_name}' not available with your current license."

        # Rate-limit for free tier advanced features
        if self._license_data.get('license_type') == 'free' and feature_name in self.advanced_features:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            if self._usage_data.get('date') != today:
                self._usage_data['date'] = today
                self._usage_data['api_calls'] = 0
                self._usage_data['advanced_operations'] = 0

            max_daily_ops = features.get('daily_advanced_operations', 1)
            current_ops = self._usage_data.get('advanced_operations', 0)
            if current_ops >= max_daily_ops:
                return False, f"Daily limit reached for advanced feature: {feature_name}. Upgrade to premium for unlimited usage."
            self._usage_data['advanced_operations'] = current_ops + 1
            if 'feature_usage' not in self._usage_data:
                self._usage_data['feature_usage'] = {}
            self._usage_data['feature_usage'][feature_name] = self._usage_data['feature_usage'].get(feature_name, 0) + 1
            self._save_usage_data()
            return True, f"Access granted to advanced feature: {feature_name}. Daily usage: {current_ops + 1}/{max_daily_ops}"

        return True, f"Access granted to feature: {feature_name}"

    def get_remaining_daily_usage(self) -> Dict[str, Any]:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if self._usage_data.get('date') != today:
            self._usage_data['date'] = today
            self._usage_data['api_calls'] = 0
            self._usage_data['advanced_operations'] = 0
            self._save_usage_data()

        license_type = self._license_data.get('license_type', 'free')
        features = self._license_data.get('features', {})
        if license_type != 'free':
            return {'limited': False, 'message': 'No usage limits for your license tier'}

        max_daily_ops = features.get('daily_advanced_operations', 1)
        current_ops = self._usage_data.get('advanced_operations', 0)
        remaining_ops = max(0, max_daily_ops - current_ops)
        max_api_calls = features.get('daily_api_calls', 5)
        current_calls = self._usage_data.get('api_calls', 0)
        remaining_calls = max(0, max_api_calls - current_calls)
        feature_usage = self._usage_data.get('feature_usage', {})
        return {
            'limited': True,
            'date': today,
            'advanced_operations': {'used': current_ops, 'limit': max_daily_ops, 'remaining': remaining_ops},
            'api_calls': {'used': current_calls, 'limit': max_api_calls, 'remaining': remaining_calls},
            'feature_usage': feature_usage,
            'resets_at': 'midnight'
        }

    def get_license_info(self) -> Dict[str, Any]:
        """Return a compact dictionary with current license information.

        This is used by the CLI to show license status. It intentionally
        returns plain serializable types (no complex objects).
        """
        try:
            info = {
                'license_key': self._license_data.get('license_key'),
                'license_type': self._license_data.get('license_type', 'free'),
                'activated': bool(self._license_data.get('activated', False)),
                'customer_id': self._license_data.get('customer_id'),
                'expires': self._license_data.get('expires'),
                'activation_date': self._license_data.get('activation_date'),
                'machine_code': self._license_data.get('machine_code'),
                'is_academic': bool(self._license_data.get('is_academic', False)),
                'features': self._license_data.get('features', {}),
            }
        except Exception:
            # Defensive fallback in case internal state is unexpected
            info = {
                'license_key': None,
                'license_type': 'free',
                'activated': False,
                'customer_id': None,
                'expires': None,
                'activation_date': None,
                'machine_code': None,
                'is_academic': False,
                'features': {},
            }

        # Include current usage summary (keeps the structure lightweight)
        try:
            info['usage'] = self.get_remaining_daily_usage()
        except Exception:
            info['usage'] = {'limited': True}

        # Backwards-compatibility: some callers expect a 'status' field
        try:
            if info.get('activated'):
                info['status'] = 'active'
            else:
                # Derive a status from license_type when not activated
                lt = info.get('license_type') or 'free'
                info['status'] = 'inactive' if lt == 'free' else lt
        except Exception:
            info['status'] = 'unknown'

        return info

    # Simple helpers that read from current license
    def get_max_datasets(self) -> int:
        return self._license_data.get('features', {}).get('max_datasets', 5)

    def get_allowed_export_formats(self) -> list:
        return self._license_data.get('features', {}).get('export_formats', ['json', 'csv', 'xlsx', 'parquet', 'html', 'md', 'sqlite'])

    def get_max_file_size_mb(self) -> float:
        return self._license_data.get('features', {}).get('max_file_size_mb', 1024)

    def get_max_concurrent_operations(self) -> int:
        return self._license_data.get('features', {}).get('concurrent_operations', float('inf'))

    def can_use_advanced_visualizations(self) -> bool:
        return self._license_data.get('features', {}).get('advanced_visualizations', False)

    def can_manage_custom_transforms(self) -> bool:
        return self._license_data.get('features', {}).get('custom_transformations', False)

    def check_file_size_limit(self, file_size_mb: float) -> Tuple[bool, str]:
        max_size = self.get_max_file_size_mb()
        if max_size == float('inf'):
            return True, 'No file size limit'
        if file_size_mb > max_size:
            license_type = self._license_data.get('license_type', 'none')
            if license_type in ['trial', 'none']:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds trial limit of {max_size}MB (1GB). Upgrade to premium for unlimited file sizes."
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit of {max_size}MB."
        return True, f"File size OK ({file_size_mb:.1f}MB <= {max_size}MB)"

    def show_upgrade_info(self) -> str:
        license_type = self._license_data.get('license_type', 'none')
        if license_type == 'free':
            usage_info = self.get_remaining_daily_usage()
            remaining_ops = usage_info['advanced_operations']['remaining']
            upgrade_message = f"""
+==============================================================================+
|                            TENSORPACK UPGRADE                                |
+==============================================================================+
|                                                                              |
| Current Status: FREE TIER                                                   |
| Advanced Operations Remaining Today: {remaining_ops}/1                                     |
|                                                                              |
| UPGRADE TO PREMIUM - £250 (One-time payment)                               |
|                                                                              |
| WHAT YOU GET:                                                               |
|   • Unlimited advanced features (traverse_graph, discover_connections)      |
|   • Custom transformation registry                                          |
|   • All export formats (CSV, Excel, Parquet, etc.)                        |
|   • Interactive 3D visualizations                                          |
|   • Unlimited file size & datasets                                         |
|   • Concurrent processing                                                   |
|   • Priority support                                                        |
|   • Lifetime updates                                                        |
|                                                                              |
| ACADEMIC USERS (.ac.uk domains):                                           |
|   • Get premium features for FREE                                          |
|   • No payment required                                                     |
|                                                                              |
| PURCHASE: {self.UPGRADE_URL}                             |
|                                                                              |
| After purchase, activate with:                                             |
|    tensorpack --activate-license YOUR-KEY                                   |
|                                                                              |
+==============================================================================+
"""
        else:
            upgrade_message = f"""
+==============================================================================+
|                          TENSORPACK LICENSE INFO                             |
+==============================================================================+
|                                                                              |
| Current Status: {license_type.upper()} LICENSE                                             |
|                                                                              |
| You have access to all premium features!                                   |
|                                                                              |
| Support: support@tensorpack.ai                                             |
| Website: {self.UPGRADE_URL}                             |
|                                                                              |
+==============================================================================+
"""
        return upgrade_message

    def open_upgrade_page(self) -> bool:
        try:
            webbrowser.open(self.UPGRADE_URL)
            print(f"Opening upgrade page in your browser...")
            print(f"URL: {self.UPGRADE_URL}")
            return True
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Please visit: {self.UPGRADE_URL}")
            return False

    def clear_license(self) -> bool:
        """Clear any locally stored license and usage data for this user.

        This removes the persisted license and usage files and resets the
        in-memory license state to the default free tier. Intended for
        developers or to allow a fresh activation per-user.
        """
        try:
            # Reset in-memory state
            self._license_data = {
                'license_key': None,
                'customer_id': None,
                'license_type': 'free',
                'activated': False,
                'expires': None,
                'activation_date': None,
                'machine_code': None,
                'features': self.feature_access.get('free', {}),
                'metadata': {}
            }

            # Reset usage
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            self._usage_data = {'date': today, 'api_calls': 0, 'advanced_operations': 0, 'feature_usage': {}}

            # Remove persisted files if present
            try:
                if self.license_file.exists():
                    self.license_file.unlink()
            except Exception:
                pass
            try:
                if self.usage_file.exists():
                    self.usage_file.unlink()
            except Exception:
                pass

            # Persist the cleared default state
            self._save_license_data()
            self._save_usage_data()
            logger.info('Local license and usage data cleared.')
            return True
        except Exception as e:
            logger.error(f'Failed to clear license data: {e}')
            return False
