# AppleSupport Annotation Guidelines

## Overview
The goal of this annotation task is to classify incoming customer tweets into a specific intent from our taxonomy, determine if it should be auto-escalated to a human agent, and rate the difficulty of the classification.

## Labels & Taxonomy

You must select EXACTLY ONE of the following intents.

1. **device_troubleshooting**
   - *Include:* Hardware issues, broken screens, battery draining fast, device not turning on.
   - *Exclude:* iOS bugs, app crashes (use `software_issue`).

2. **software_issue**
   - *Include:* iOS updates failing, specific apps crashing, settings not saving, bugs.
   - *Exclude:* Forgotten passwords (use `account_access`).

3. **account_access**
   - *Include:* Forgotten Apple ID password, locked accounts, 2FA issues.
   - *Exclude:* iCloud storage full (use `subscription_management`).

4. **payment_billing**
   - *Include:* Unknown charges, refund requests, Apple Card issues.
   - *Exclude:* Ordering a new physical device (use `purchase_order`).

5. **subscription_management**
   - *Include:* Apple Music, Apple TV+, iCloud storage upgrades/cancellations.

6. **repair_status**
   - *Include:* "Where is my phone?", "Is my repair done?", AppleCare+ claim status.

7. **purchase_order**
   - *Include:* Questions about shipping times for new devices, changing an order.

8. **information_request**
   - *Include:* "How do I take a screenshot?", "Is the iPhone 15 waterproof?"

9. **complaint**
   - *Include:* Rants, threats to switch to Android, severe dissatisfaction without a clear technical question.

10. **other**
    - *Include:* Spam, unintelligible messages, praise/compliments, or anything not covered above.

## `should_escalate` (Boolean)
Set to `TRUE` if:
- It involves sensitive PII, account recovery, or billing disputes.
- It is a severe complaint.
- The user explicitly demands to speak to a human.

Set to `FALSE` if:
- It is a standard troubleshooting question or how-to request that could safely be answered by an AI referencing a historical KB.

## `difficulty` (String)
- **Easy:** Obvious intent, clear language.
- **Medium:** Contains multiple questions, but one primary intent dominates.
- **Hard:** Very vague, borderline between two intents (e.g., software issue causing battery drain), or uses heavy slang/sarcasm.

## Handling Edge Cases
- If a tweet contains multiple intents (e.g., "My screen is cracked and I forgot my password"), choose the intent that is **most critical** (e.g., `account_access` over `device_troubleshooting`).
- If you are completely unsure, label as `other` and add notes in the `annotation_notes` column. Do NOT guess blindly.
