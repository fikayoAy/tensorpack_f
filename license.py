import jwt  # PyJWT
import datetime

# Load private key
with open("private_key.pem", "rb") as f:
    private_key = f.read()

# License payload
payload = {
    "user": "ayodeleanjola4@gmail.com",
    "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),  # expires in 30 days
    "features": ["pro", "analysis"]
}

# Create signed JWT
token = jwt.encode(payload, private_key, algorithm="RS256")

# Save license file
with open("alice_license.jwt", "w") as f:
    f.write(token)

print("License created for Alice!")
