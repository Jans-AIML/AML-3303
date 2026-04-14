"""
Create Slack Webhook

- Go to your Slack workspace.
- Install the app → Incoming Webhooks
- Create a new webhook, choose a channel (#mlalerts).
- Copy the Webhook URL (you’ll need this in code).
"""

## Reference Code

import requests
import time

# Replace with your actual webhook URL
WEBHOOK_URL = "https://hooks.slack.com/services/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"

def send_slack_message(message):
    payload = {"text": message}
    requests.post(WEBHOOK_URL, json=payload)

# Example: Simulate training alerts
for epoch in range(1, 6):
    accuracy = 0.7 + epoch * 0.05
    msg = f"Training Update - Epoch {epoch}: Accuracy = {accuracy:.2f}"
    send_slack_message(msg)
    time.sleep(2)  # simulate processing time

send_slack_message("✅ Training Completed!")
