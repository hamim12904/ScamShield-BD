"""Hand-written hard-legit and minimal-pair examples for A5.

Written from the competition definitions and labeled sample patterns only; no test rows were read or copied. Each six-row group shares a `group_id` so folds never split a minimal pair.
"""
import pandas as pd

VALID_TECHNIQUES = {"T1", "T2", "T3", "T4", "T5", "T6"}
COLUMNS = ["id", "text", "label", "techniques", "action", "group_id", "source", "weight", "family", "language"]


def _tech(*codes):
    if not codes:
        return "NONE"
    return "|".join(sorted(codes))


ROWS = [
    (90001, 9001, "bKash support: your account will be paused tomorrow. Reply with the 6-digit code we just sent to confirm you are the owner.", 1, _tech("T1", "T2", "T5"), 1, "otp", "en"),
    (90002, 9001, "Nagad helpline: apnar account e sobchhe chole gese. KYC update korte PIN ar OTP ta ei number e pathan, na hole block hobe.", 1, _tech("T1", "T2", "T5"), 1, "otp", "banglish"),
    (90003, 9001, "S: We detected a login on your wallet ||| R: I am not at home ||| S: Then share the code we sent to keep access", 1, _tech("T1", "T2", "T5"), 1, "otp", "en"),
    (90004, 9001, "bKash: Your new device login was approved at 18:42. If this was not you, call 16247 from the app. Never share your OTP with anyone, including our staff.", 0, _tech(), 0, "otp", "en"),
    (90005, 9001, "বিকাশ: আপনার অ্যাকাউন্টে নতুন ডিভাইস যোগ করা হয়েছে। যদি আপনি না করেন, অ্যাপ থেকে ১৬২৪৭ কল করুন। কাউকে ওটিপি বা পিন দেবেন না।", 0, _tech(), 0, "otp", "bn"),
    (90006, 9001, "S: bKash support here. We will never ask for your OTP. ||| R: I got a code, should I read it? ||| S: No. Keep it private and report the message in the app.", 0, _tech(), 0, "otp", "en"),

    (90007, 9002, "Daraz refund desk: Tk 1,250 refund approved. Send card number, expiry, CVV and OTP to release it.", 1, _tech("T1", "T2", "T5"), 1, "refund", "en"),
    (90008, 9002, "Bhai, bhul kore apnar number e 5,000 taka gese. Ferot pathat hole bKash PIN ta bolen.", 1, _tech("T2", "T5"), 1, "refund", "banglish"),
    (90009, 9002, "S: I sent money to the wrong number by mistake ||| R: I did not receive it ||| S: Please check and send it back; verify with your PIN first", 1, _tech("T2", "T5"), 1, "refund", "en"),
    (90010, 9002, "bKash: Tk 1,250 refund from Daraz was credited to your wallet. TrxID BX91K7. No action needed; we never ask for PIN or OTP.", 0, _tech(), 0, "refund", "en"),
    (90011, 9002, "আপনার বিকাশ ওয়ালেটে ১,২৫০ টাকা রিফান্ড জমা হয়েছে। ট্রানজেকশন আইডি BX91K7। কোনো পিন বা ওটিপি দিতে হবে না।", 0, _tech(), 0, "refund", "bn"),
    (90012, 9002, "S: Did you receive the wrong-number transfer? ||| R: Yes, I will return it through bKash myself. ||| S: Good. Never ask anyone for your PIN to fix it.", 0, _tech(), 0, "refund", "en"),

    (90013, 9003, "SA Courier: parcel CX8821 stuck at customs. Pay Tk 450 clearance fee at hxxps://sa-clearing[.]top before 6 pm.", 4, _tech("T1", "T3", "T5"), 2, "delivery", "en"),
    (90014, 9003, "আপনার পার্সেল আটকে আছে। আজই ৫০ টাকা ডেলিভারি ফি দিতে লিংকে ক্লিক করুন: hxxps://parcel-fee[.]top", 4, _tech("T1", "T3", "T5"), 2, "delivery", "bn"),
    (90015, 9003, "S: DHL here, customs hold ||| R: I did not order anything ||| S: Pay Tk 900 duty today or the parcel returns", 4, _tech("T1", "T3", "T5"), 2, "delivery", "en"),
    (90016, 9003, "Pathao Parcel: delivery attempt 2 for tracking BD4471. Reschedule from the official app. Our riders never ask for advance payment by SMS link.", 0, _tech(), 0, "delivery", "en"),
    (90017, 9003, "পাঠাও পার্সেল: আপনার BD4471 প্যারাসেল আজ বিকেল ৪টায় ডেলিভারি হবে। সময় বদলাতে অ্যাপ ব্যবহার করুন; এসএমএস লিংকে টাকা দেবেন না।", 0, _tech(), 0, "delivery", "bn"),
    (90018, 9003, "S: Courier text says pay a fee via a link ||| R: Did you check the tracking number in the app? ||| S: Yes, it is already out for delivery. The link was fake.", 0, _tech(), 0, "delivery", "en"),

    (90019, 9004, "Lucky draw: your number won a Tk 50,000 voucher. Claim at hxxps://win-bd[.]top within 60 minutes.", 2, _tech("T4", "T5"), 3, "prize", "en"),
    (90020, 9004, "Oi apni 10 lakh taka jitechen. Claim korte age 300 tk processing fee pathan.", 2, _tech("T3", "T4", "T5"), 2, "prize", "banglish"),
    (90021, 9004, "S: Congrats, you won a smartphone ||| R: really? ||| S: Pay Tk 500 delivery charge and confirm your address", 2, _tech("T3", "T4"), 2, "prize", "en"),
    (90022, 9004, "Robi: EID bonus 2 GB added to your number. Check balance with *121*3#. No claim link or fee is required.", 5, _tech(), 0, "prize", "en"),
    (90023, 9004, "জিপির ঈদ অফার: ১ জিবি কিনলে ১ জিবি ফ্রি। *১২১*৩# ডায়াল করে দেখুন; কোনো ফি বা লিংক নেই।", 5, _tech(), 0, "prize", "bn"),
    (90024, 9004, "S: Is the cashback message real? ||| R: I checked the operator app, it is a genuine 10% top-up cashback. ||| S: Good, no need to click the SMS link.", 5, _tech(), 0, "prize", "en"),

    (90025, 9005, "Part-time task: rate 5 apps daily, earn Tk 800. Join hxxps://t[.]me/task_bd and deposit Tk 2,000 as security.", 3, _tech("T3", "T6"), 2, "job", "en"),
    (90026, 9005, "Freelance job: protidin 3000 tk income. WhatsApp e join korte fee 1500 tk pathan.", 3, _tech("T3", "T6"), 2, "job", "banglish"),
    (90027, 9005, "S: We saw your CV, data entry role ||| R: how much? ||| S: Tk 40,000 monthly, pay Tk 3,000 training fee first", 3, _tech("T1", "T3", "T6"), 2, "job", "en"),
    (90028, 9005, "Bdjobs: Your application for Data Entry Officer was received. Employer will contact you through Bdjobs messages only. Never pay to get a job.", 0, _tech(), 0, "job", "en"),
    (90029, 9005, "বিডিজবস: আপনার আবেদন গৃহীত হয়েছে। নিয়োগকর্তা শুধু বিডিজবস মেসেজে যোগাযোগ করবেন। টাকা দিয়ে চাকরি কেনা যাবে না।", 0, _tech(), 0, "job", "bn"),
    (90030, 9005, "S: Recruiter asking for a training fee ||| R: That is a scam; real employers never charge you ||| S: Thanks, I reported the chat.", 0, _tech(), 0, "job", "en"),

    (90031, 9006, "Invest Tk 10,000 in our crypto bot, get 25% weekly profit guaranteed. Send to bKash 01XXXXXXXXX now.", 3, _tech("T3", "T5", "T6"), 2, "investment", "en"),
    (90032, 9006, "Ami investment group e 300% profit dekhechi. Join korte Tk 5,000 fee dite hobe.", 3, _tech("T3", "T6"), 4, "investment", "banglish"),
    (90033, 9006, "S: Sir, double your money in 7 days ||| R: is it safe? ||| S: 100% guaranteed, transfer today only", 3, _tech("T3", "T5", "T6"), 2, "investment", "en"),
    (90034, 9006, "Bangladesh Bank advisory: verify any investment offer against the licensed list. High 'guaranteed' returns are a warning sign.", 0, _tech(), 0, "investment", "en"),
    (90035, 9006, "বাংলাদেশ ব্যাংক সতর্কতা: বিনিয়োগের আগে লাইসেন্স তালিকা যাচাই করুন। 'নিশ্চিত লাভ' প্রতারণার লক্ষণ হতে পারে।", 0, _tech(), 0, "investment", "bn"),
    (90036, 9006, "S: This investment chat promises guaranteed profit ||| R: Do not transfer; check the regulator list first ||| S: Agreed, I will report it.", 0, _tech(), 0, "investment", "en"),

    (90037, 9007, "Robi: your SIM will be blocked tomorrow for unverified KYC. Reply with NID number and OTP to keep it active.", 1, _tech("T1", "T2", "T5"), 1, "bill", "en"),
    (90038, 9007, "নগদ: অ্যাকাউন্ট বন্ধ হতে পারে। এখনই পিন নম্বর পাঠান।", 1, _tech("T1", "T2", "T5"), 1, "bill", "bn"),
    (90039, 9007, "S: Telecom support here ||| R: why? ||| S: Confirm your SIM by sharing the OTP we just sent", 1, _tech("T1", "T2", "T5"), 1, "bill", "en"),
    (90040, 9007, "Titas Gas bill reminder: Tk 860 due on 30 Nov. Pay through the official app or bank branch. Ignore SMS links asking for OTP.", 0, _tech(), 0, "bill", "en"),
    (90041, 9007, "টিটাস গ্যাস: আপনার বিল ৮৬০ টাকা ৩০ নভেম্বরের আগে দিতে হবে। অফিশিয়াল অ্যাপ বা ব্যাংক ব্যবহার করুন; ওটিপি চাওয়া বার্তা এড়িয়ে চলুন।", 0, _tech(), 0, "bill", "bn"),
    (90042, 9007, "S: Gas bill text says pay via link ||| R: I paid in the app; the link was not official ||| S: Good, never share OTP for bill payment.", 0, _tech(), 0, "bill", "en"),

    (90043, 9008, "Nagad app update required. Download hxxps://nagad-update[.]top/app.apk and enter old PIN to keep wallet active.", 1, _tech("T1", "T2", "T4", "T5"), 4, "service_link", "en"),
    (90044, 9008, "Voucher activate korte app install korun hxxps://gp-voucher[.]top/a.apk ar PIN din.", 1, _tech("T1", "T2", "T4"), 4, "service_link", "banglish"),
    (90045, 9008, "S: Support says install an update ||| R: from where? ||| S: this link, then share your old PIN", 1, _tech("T1", "T2", "T4"), 4, "service_link", "en"),
    (90046, 9008, "bKash: Scheduled maintenance tonight 01:00-03:00. Details: hxxps://bkash[.]com/notice. No app download or PIN entry is needed.", 0, _tech(), 0, "service_link", "en"),
    (90047, 9008, "বিকাশ: আজ রাত ১টা থেকে ৩টা রক্ষণাবেক্ষণ। বিবরণ hxxps://bkash[.]com/notice; কোনো পিন বা অ্যাপ ইনস্টলের প্রয়োজন নেই।", 0, _tech(), 0, "service_link", "bn"),
    (90048, 9008, "S: Maintenance notice has a link ||| R: I opened the official domain, not the forwarded one ||| S: Correct; never install apps from SMS links.", 0, _tech(), 0, "service_link", "en"),

    (90049, 9009, "Mama, ami bipode. Amar bKash e 3,000 taka nai. Tomar PIN ta dao, ami send kore nei.", 1, _tech("T2", "T5"), 1, "friend_warning", "banglish"),
    (90050, 9009, "Bhai, ami emergency e achhi. OTP ta patha diben, amar account block hoye jabe.", 1, _tech("T2", "T5"), 1, "friend_warning", "banglish"),
    (90051, 9009, "S: It is me, new number ||| R: really? ||| S: yes, send the OTP I just texted you", 1, _tech("T2", "T5"), 1, "friend_warning", "en"),
    (90052, 9009, "Ami ekta fake bKash message dekhechi. Kauke OTP ba PIN bolben na; ami amar number change korini.", 0, _tech(), 0, "friend_warning", "banglish"),
    (90053, 9009, "আমি একটা জাল bKash বার্তা দেখেছি। কাউকে ওটিপি বা পিন বলবেন না; আমি আমার নম্বর পরিবর্তন করিনি।", 0, _tech(), 0, "friend_warning", "bn"),
    (90054, 9009, "S: Bro, someone texted from my number asking for OTP ||| R: I did not share it ||| S: Good, block and report that number.", 0, _tech(), 0, "friend_warning", "en"),

    (90055, 9010, "You won Tk 7,000 cashback. Confirm by clicking hxxps://cb-bd[.]top and pay Tk 250 verification fee.", 2, _tech("T3", "T4", "T5"), 2, "cashback", "en"),
    (90056, 9010, "Cashback 5,000 tk jitechen. Link e click kore address confirm korun.", 2, _tech("T4", "T5"), 3, "cashback", "banglish"),
    (90057, 9010, "S: Congrats, cashback winner ||| R: wow ||| S: Click this link and pay Tk 200 processing", 2, _tech("T3", "T4"), 3, "cashback", "en"),
    (90058, 9010, "Aarong EID offer: 20% off on all dresses this week. Valid in store and on the official app; no fee or link needed.", 5, _tech(), 0, "cashback", "en"),
    (90059, 9010, "আরংগ ঈদ অফার: সব পোশাকে ২০% ছাড়। দোকান ও অফিশিয়াল অ্যাপে কার্যকর; কোনো ফি বা লিংক নেই।", 5, _tech(), 0, "cashback", "bn"),
    (90060, 9010, "S: Is this cashback real? ||| R: I checked the official app, yes, 10% top-up cashback. ||| S: Great, no need to reply to the SMS.", 5, _tech(), 0, "cashback", "en"),

    (90061, 9011, "Courier tracking BD9911 failed. Confirm address at hxxps://bd-track[.]top; pay Tk 60 redelivery now.", 4, _tech("T1", "T3", "T5"), 2, "tracking", "en"),
    (90062, 9011, "আপনার পার্সেলের ঠিকানা ভুল। hxxps://bd-track[.]top এ ক্লিক করে ৬০ টাকা দিন।", 4, _tech("T1", "T3"), 3, "tracking", "bn"),
    (90063, 9011, "S: Courier here ||| R: I have no parcel ||| S: It is in your name, pay Tk 60 customs today", 4, _tech("T1", "T3", "T5"), 2, "tracking", "en"),
    (90064, 9011, "Sundarban Courier: tracking BD9911 out for delivery today. Check status on the official website; no payment requested.", 0, _tech(), 0, "tracking", "en"),
    (90065, 9011, "সুন্দরবন কুরিয়ার: BD9911 আজ ডেলিভারিতে আছে। অফিশিয়াল ওয়েবসাইটে দেখুন; কোনো টাকা চাওয়া হয়নি।", 0, _tech(), 0, "tracking", "bn"),
    (90066, 9011, "S: Delivery message asks for a fee ||| R: Official tracking says out for delivery ||| S: Then it is phishing; ignore the link.", 0, _tech(), 0, "tracking", "en"),

    (90067, 9012, "HR: your salary is on hold. Share bank account and OTP to release Tk 45,000 today.", 1, _tech("T1", "T2", "T5"), 1, "payroll", "en"),
    (90068, 9012, "HR office: salary release korte account details ar OTP pathan.", 1, _tech("T1", "T2"), 1, "payroll", "banglish"),
    (90069, 9012, "S: Payroll team here ||| R: why OTP? ||| S: to verify salary account, send now", 1, _tech("T1", "T2", "T5"), 1, "payroll", "en"),
    (90070, 9012, "Payroll: salary Tk 45,000 credited to account ending 4412. Payslip is in the office portal. HR never asks for OTP.", 0, _tech(), 0, "payroll", "en"),
    (90071, 9012, "বেতন: ৪৫,০০০ টাকা ৪৪১২ নম্বর হিসাব জমা হয়েছে। পেসলিপ অফিস পোর্টালে আছে; এইচআর কখনো ওটিপি চায় না।", 0, _tech(), 0, "payroll", "bn"),
    (90072, 9012, "S: HR text asks for bank OTP ||| R: Never share it; I checked the portal ||| S: Reported to security.", 0, _tech(), 0, "payroll", "en"),

    (90073, 9013, "Customer loyalty survey: you qualified for Tk 15,000 gift. Pay Tk 100 handling fee at hxxps://survey-bd[.]top to claim.", 2, _tech("T1", "T3", "T4"), 2, "survey", "en"),
    (90074, 9013, "Survey e 15,000 tk gift. Claim korte hxxps://survey-bd[.]top e click korun.", 2, _tech("T4", "T5"), 3, "survey", "banglish"),
    (90075, 9013, "S: Loyalty reward selected ||| R: what? ||| S: Pay Tk 100 shipping for your gift", 2, _tech("T3", "T4"), 2, "survey", "en"),
    (90076, 9013, "Customer survey: tell us about network quality in the app and get 500 bonus points. No payment or personal data required.", 5, _tech(), 0, "survey", "en"),
    (90077, 9013, "কাস্টমার সার্ভে: অ্যাপে নেটওয়ার্ক মতামত দিন ৫০০ বোনাস পয়েন্ট পাবেন। কোনো টাকা বা ব্যক্তিগত তথ্য লাগবে না।", 5, _tech(), 0, "survey", "bn"),
    (90078, 9013, "S: Survey offers bonus points ||| R: I filled it in the official app ||| S: Good, no link or payment needed.", 5, _tech(), 0, "survey", "en"),

    (90079, 9014, "bKash support: apnar РIN expire today. Reply with currentP1N and new PIN to continue.", 1, _tech("T1", "T2", "T5"), 1, "mixed", "mixed"),
    (90080, 9014, "N0gad helpline: ver1fy account, send 0TP + P1N nowwww.", 1, _tech("T1", "T2", "T5"), 1, "mixed", "mixed"),
    (90081, 9014, "S: Assalamu alaikum, bKash agent ||| R: kotha bolun ||| S: apnar account verify korte PIN bolen ||| R: no ||| S: na hole SIM block", 1, _tech("T1", "T2", "T5"), 1, "mixed", "mixed"),
    (90082, 9014, "bKash notice: your PIN was changed today at 14:10. If not you, call 16247. We will never ask for OTP in chat.", 0, _tech(), 0, "mixed", "en"),
    (90083, 9014, "বিকাশ নোটিশ: আজ ১৪:১০ এ পিন বদল হয়েছে। আপনি না করলে ১৬২৪৭ কল করুন। চ্যাটে কখনো ওটিপি চাইব না।", 0, _tech(), 0, "mixed", "bn"),
    (90084, 9014, "S: Message says SIM will block ||| R: I checked in app, nothing pending ||| S: Good, delete it and never send OTP.", 0, _tech(), 0, "mixed", "en"),
]


def validate():
    """Check schema and label/technique/action consistency; return True."""
    assert len(ROWS) == 84, f"expected 84 rows, got {len(ROWS)}"
    ids = [r[0] for r in ROWS]
    assert len(ids) == len(set(ids)), "duplicate handwritten id"
    assert min(ids) >= 90001, "handwritten ids must not overlap sample/test ids"
    groups = {}
    for row in ROWS:
        _id, group, text, label, techniques, action, family, language = row
        groups.setdefault(group, []).append(_id)
        assert isinstance(text, str) and text.strip(), f"empty text at {_id}"
        assert text.count("|||") <= 11, f"too many messages at {_id}"
        assert label in range(6), f"bad label at {_id}"
        assert action in range(6), f"bad action at {_id}"
        assert family and language, f"missing metadata at {_id}"
        codes = [] if techniques == "NONE" else techniques.split("|")
        assert techniques == "NONE" or all(c in VALID_TECHNIQUES for c in codes), f"bad technique at {_id}"
        if techniques != "NONE":
            assert techniques == "|".join(sorted(codes)), f"techniques not sorted at {_id}"
        if label in (0, 5):
            assert techniques == "NONE" and action == 0, f"non-scam row has suspicious fields at {_id}"
        else:
            assert techniques != "NONE", f"scam row without technique at {_id}"
    assert len(groups) == 14, f"expected 14 groups, got {len(groups)}"
    assert all(len(v) == 6 for v in groups.values()), "each group must have 6 rows"
    return True


def load_handwritten(weight=0.75):
    """Return a DataFrame in the training schema with group/source/weight columns."""
    validate()
    df = pd.DataFrame(ROWS, columns=["id", "group_id", "text", "label", "techniques", "action", "family", "language"])
    df["source"] = "handwritten"
    df["weight"] = float(weight)
    return df[COLUMNS]


if __name__ == "__main__":
    out = load_handwritten()
    print(f"Validated {len(out)} handwritten rows in {out['group_id'].nunique()} groups")
    print(out["label"].value_counts().sort_index().to_dict())
