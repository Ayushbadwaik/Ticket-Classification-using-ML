import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# 1. Training Dataset
# ----------------------------
categories = {
    "Billing": [
        "I was charged twice for my subscription",
        "Incorrect billing amount shown on my invoice",
        "Why am I being billed for services I didn’t use?",
        "Refund has not been processed to my account",
        "Billing cycle is incorrect, please update",
        "Received invoice but payment already made",
        "Charged extra fees without explanation",
        "Why is my monthly bill higher than last month?",
        "Payment deducted but subscription still inactive",
        "Need breakdown of my charges this month",
        "Tax calculation on my bill seems wrong",
        "Refund taking too long, please expedite",
        "Wrong billing address on my invoice",
        "Why am I charged after cancellation?",
        "Need GST invoice for my purchase",
        "Invoice not generated this month",
        "Late fee applied wrongly to my account",
        "I upgraded plan but still charged old rate",
        "Billing history not showing in dashboard",
        "Charged twice for the same order",
        "Overdraft charges applied incorrectly",
        "Annual subscription billed monthly",
        "Extra transaction charges applied",
        "Duplicate invoice generated",
        "Discount coupon not applied to bill",
        "Payment successful but bill pending",
        "Why two different amounts are deducted?",
        "Wrong plan billed to me",
        "Need invoice copy urgently",
        "Bill amount fluctuates every month",
        "Refund not reflected in statement",
        "Charged for cancelled service",
        "Payment declined but money deducted",
        "Wrong late fee applied",
        "Invoice shows wrong due date",
        "Need correction in my invoice",
        "Plan downgraded but billed at higher rate",
        "Subscription cancelled but billed again",
        "Incorrect credit applied",
        "Bill payment pending despite transaction",
        "Charged before trial ended",
        "Unable to download invoice",
        "Bill shows items not purchased",
        "Overcharged for one-time purchase",
        "Why am I billed international fees?",
        "Duplicate billing notification received",
        "Charges not explained in invoice",
        "Bill not updated after refund",
        "Need official bill for reimbursement",
        "My bill doesn’t match my usage"
    ],
    "Technical Support": [
        "App keeps crashing after update",
        "Unable to login despite correct credentials",
        "Website not loading properly",
        "Error code 504 when making payment",
        "App stuck on loading screen",
        "Two-factor authentication not working",
        "Password reset email not received",
        "Page freezes when I click submit",
        "Update installed but features not working",
        "Error while uploading documents",
        "System runs too slow after update",
        "Screen goes blank randomly",
        "App force closes on opening",
        "Payment gateway not loading",
        "Captcha not displaying properly",
        "App not compatible with my device",
        "Error message when starting chat",
        "Microphone not detected by app",
        "Camera permissions not working",
        "App not syncing across devices",
        "Notifications not showing up",
        "Downloaded files not opening",
        "Settings not saving in app",
        "Error connecting to server",
        "Frequent disconnections in chat",
        "VPN blocking access to app",
        "Error in OTP verification",
        "Unable to update profile picture",
        "App crashing on Android 14",
        "App keeps asking for permissions",
        "Login page not loading fully",
        "Cannot connect to database",
        "Blue screen error after installation",
        "Website forms not submitting",
        "Blank page after login",
        "Slow response from server",
        "App closes after splash screen",
        "App unable to fetch data",
        "Glitches while scrolling page",
        "Error in downloading updates",
        "Sound not working in app",
        "Bluetooth not connecting to app",
        "App not starting at all",
        "Error while verifying details",
        "Dark mode not working",
        "Push notifications delayed",
        "App logout issue automatically",
        "Frequent update errors",
        "Video not loading properly",
        "Website shows broken layout"
    ],
    "Account Management": [
        "Unable to change my email address",
        "Password reset not working",
        "Want to delete my account permanently",
        "Unable to update phone number",
        "Account locked after failed attempts",
        "Want to deactivate account temporarily",
        "Unable to link my social media accounts",
        "Change username option not available",
        "Cannot update billing details",
        "Need to merge two accounts",
        "Forgot security questions",
        "Account suspended without notice",
        "Unable to recover deleted account",
        "Cannot set up two-factor authentication",
        "Email verification link expired",
        "Account details not updating",
        "Want to switch to corporate account",
        "Account hacked, need recovery",
        "Unable to unlink device from account",
        "Old account still active after closure",
        "Want to change default language",
        "Cannot update payment preferences",
        "Subscription plan not showing in account",
        "Email not linked with account",
        "Multiple accounts created mistakenly",
        "Cannot reset username",
        "Child account restrictions not working",
        "Unable to update security settings",
        "Account recovery email not received",
        "Password change not reflecting",
        "Account locked due to suspicious login",
        "Need to verify my account identity",
        "Cannot enable biometric login",
        "Want to transfer account ownership",
        "Deleted account still active",
        "Subscription linked to wrong account",
        "Cannot access account dashboard",
        "Email already registered error",
        "Phone number update not saving",
        "Account deletion taking too long",
        "Want to reactivate account",
        "Security code not working",
        "Linked accounts not showing",
        "Subscription not linked properly",
        "Account preferences reset automatically",
        "Unable to logout of account",
        "Cannot disable auto-login",
        "Need activity log for my account",
        "Old email still linked to account",
        "Want to change country settings"
    ]
}

# Convert to dataset
data = {"ticket_text": [], "category": []}
for category, texts in categories.items():
    data["ticket_text"].extend(texts)
    data["category"].extend([category] * len(texts))

# Create DataFrame
df = pd.DataFrame(data)
print("Dataset size:", df.shape)

# ----------------------------
# 2. Vectorization
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ticket_text"])
y = df["category"]

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 4. Train Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# 5. Evaluate
# ----------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 6. Save Model and Vectorizer
# ----------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")
