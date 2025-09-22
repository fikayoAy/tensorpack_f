"""
Simplified LicenseManager using JWT for license verification.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import json
import datetime
import os
import webbrowser
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

# PayPal payment link (user requested specific link for purchases)
PAYPAL_PAYMENT_LINK = "https://www.paypal.com/ncp/payment/8W7H55GYX66X8"

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('tensorpack.license')


class LicenseManager:
    """License management using JWT for verification.

    Behaviour:
    - Verifies a license using a JWT token.
    - The public key for verification is stored in `public_key.pem`.
    - Falls back to cached license data if offline.
    """

    def __init__(self):
        # Product identifier used in payloads and messages
        self.PRODUCT_ID = "tensorpack-premium"

        # Academic domains that get premium features automatically
        self.academic_domains = [".ac.uk"]

        # Pricing and trial settings
        self.STANDARD_PRICE = 850.00
        self.TRIAL_DAYS = 14

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
                "max_datasets": 5,
                "max_file_size_mb": 50,
                "concurrent_operations": 2,
                "daily_api_calls": 5,
                "daily_advanced_operations": 5
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
        
        # Load public key for JWT verification (embedded directly in code)
        self.public_key = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAipoqo4mQXJFjwnmUoK0H
jwbx2FxeHSNR1jKZFkCEU6rnedEr0sUVBlSN/BVcSU84JwnoedldGQW8u2mK2qcZ
23l2qWanleFrTvh+h3RJxIbLAvVJDqzdo2oyDEdQx6bS/UnNFO22U+LB/+m6kqd4
CocwJVR8cpAh+Cqjb34BIPKhDber35jm4f+OMLb8Z+7oh2rcv+rUwtn93eKLKvCz
RjkL5Ubk2DJk0KGaO89cYJ1T4RB9aYxvAu6+oycCySKqrn/u7pjoUcMJgWptHGmG
2eTY0yLxX4ixok6kyyWIKjnfStjcA++sTioI+y1Sm141g/SRXSPkDmpNThoPlMCO
NwIDAQAB
-----END PUBLIC KEY-----"""

        # Load cached data
        self._license_data = self._load_license_data()
        self._usage_data = self._load_usage_data()

        # Advanced features list
        self.advanced_features = ["traverse_graph", "discover_connections", "custom_transformations", "list_transforms", "describe_transforms", "remove_transforms"]

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

    # ----------------------- Activation & processing ----------------------
    def activate_license(self, license_key: str) -> Tuple[bool, str]:
        """Activate using a JWT license key."""
        if not self.public_key:
            return False, "Public key not loaded, cannot verify license."

        # Debug logging
        logger.debug(f"Activating license with key length: {len(license_key)}")
        logger.debug(f"License key segments: {len(license_key.split('.'))}")
        logger.debug(f"First 50 chars: {license_key[:50]}...")
        
        try:
            payload = jwt.decode(license_key, self.public_key, algorithms=["RS256"])
            
            license_type = payload.get('license_type', 'premium')
            email = payload.get('user')
            is_academic = self._is_academic_email(email) if email else False

            if is_academic:
                license_type = 'academic'

            features = self.feature_access.get(license_type, self.feature_access['free'])

            self._license_data = {
                'license_key': license_key,
                'customer_id': email,
                'license_type': license_type,
                'activated': True,
                'activation_date': datetime.datetime.now().isoformat(),
                'expires': datetime.datetime.fromtimestamp(payload['exp']).isoformat() if 'exp' in payload else None,
                'machine_code': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME'),
                'features': features,
                'is_academic': is_academic,
                'metadata': {'user': payload.get('user')}
            }
            self._save_license_data()
            logger.info(f"License activated for: {payload.get('user')}")
            return True, f"License valid for: {payload.get('user')}"

        except ExpiredSignatureError:
            logger.warning("Attempted to activate an expired license.")
            return False, "License expired."
        except InvalidTokenError as e:
            logger.error(f"Invalid license token: {e}")
            return False, "Invalid license."

    # ----------------------- Verification ---------------------------------
    def verify_license(self) -> Tuple[bool, str]:
        """Verify currently stored license."""
        if self._license_data.get('license_type') == 'free':
            return True, 'Free tier license is valid.'

        license_key = self._license_data.get('license_key')
        if not license_key:
            return False, 'No license found. Please activate a license.'
        
        if not self.public_key:
            return False, "Public key not loaded, cannot verify license."

        try:
            payload = jwt.decode(license_key, self.public_key, algorithms=["RS256"])
            
            # Re-apply license data on verification to ensure it's fresh
            license_type = payload.get('license_type', 'premium')
            email = payload.get('user')
            is_academic = self._is_academic_email(email) if email else False

            if is_academic:
                license_type = 'academic'
            
            features = self.feature_access.get(license_type, self.feature_access['free'])

            self._license_data.update({
                'customer_id': email,
                'license_type': license_type,
                'activated': True,
                'expires': datetime.datetime.fromtimestamp(payload['exp']).isoformat() if 'exp' in payload else None,
                'features': features,
                'is_academic': is_academic,
            })
            self._save_license_data()
            return True, f"License verified for: {payload.get('user')}"

        except ExpiredSignatureError:
            self._license_data.update({
                'license_type': 'free',
                'features': self.feature_access['free'],
                'activated': False,
            })
            self._save_license_data()
            return False, "License expired. Downgraded to free."
        except InvalidTokenError:
            self._license_data.update({
                'license_type': 'free',
                'features': self.feature_access['free'],
                'activated': False,
            })
            self._save_license_data()
            return False, "Invalid license. Downgraded to free."

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

            max_daily_ops = features.get('daily_advanced_operations', 5)
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
        if license_type != 'free' and license_type != 'none':
            return {'limited': False, 'message': 'No usage limits for your license tier'}
        max_daily_ops = features.get('daily_advanced_operations', 5)
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
| UPGRADE TO PREMIUM - £850 (One-time payment)                               |
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
