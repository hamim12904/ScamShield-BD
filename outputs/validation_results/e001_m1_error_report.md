# Error report: e001_m1

## Confusion matrix (true rows, predicted columns)

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 Legitimate | 28 | 1 | 0 | 1 | 1 | 1 |
| 1 MFS/OTP scam | 4 | 7 | 1 | 0 | 0 | 0 |
| 2 Prize/lottery scam | 3 | 2 | 4 | 0 | 0 | 1 |
| 3 Fake job/investment | 4 | 1 | 0 | 4 | 0 | 2 |
| 4 Delivery/parcel phishing | 4 | 0 | 1 | 0 | 4 | 0 |
| 5 Promotional spam | 10 | 0 | 0 | 0 | 0 | 2 |

![confusion](e001_m1_confusion.png)

## Per-class metrics

| label | class | precision | recall | f1 | support |
|---|---|---|---|---|---|
| 0 | Legitimate | 0.5283 | 0.8750 | 0.6588 | 32 |
| 1 | MFS/OTP scam | 0.6364 | 0.5833 | 0.6087 | 12 |
| 2 | Prize/lottery scam | 0.6667 | 0.4000 | 0.5000 | 10 |
| 3 | Fake job/investment | 0.8000 | 0.3636 | 0.5000 | 11 |
| 4 | Delivery/parcel phishing | 0.8000 | 0.4444 | 0.5714 | 9 |
| 5 | Promotional spam | 0.3333 | 0.1667 | 0.2222 | 12 |
| macro |  | 0.6274 | 0.4722 | 0.5102 | 86 |

## Slices (Task A macro-F1)

| slice | macro_f1 |
|---|---|
| lang_bangla_script | 0.0556 |
| lang_banglish | 0.2900 |
| lang_english | 0.4880 |
| slice_hard_legit | 0.4375 |
| struct_conversation | 0.2667 |
| struct_single | 0.5259 |

## Per-technique metrics (thresholded predictions)

| technique | precision | recall | f1 | support | predicted |
|---|---|---|---|---|---|
| T1 | 0.4444 | 0.1905 | 0.2667 | 21 | 9 |
| T2 | 0.5000 | 0.0909 | 0.1538 | 11 | 2 |
| T3 | 0.5000 | 0.1538 | 0.2353 | 13 | 4 |
| T4 | 1.0000 | 0.2500 | 0.4000 | 12 | 3 |
| T5 | 0.6667 | 0.2222 | 0.3333 | 18 | 6 |
| T6 | 1.0000 | 0.1818 | 0.3077 | 11 | 2 |
| macro | 0.6852 | 0.1815 | 0.2828 | 86 | 26 |

## Legitimate predicted as scam

| id | true | pred | confidence | text |
|---|---|---|---|---|
| 10051 | 0 | 1 | 0.491 | Ami ekhon bkash e 500 taka pathaisi, dekh to ashche kina. |
| 10021 | 0 | 4 | 0.376 | Sundarban Courier: Parcel SC77124 arrived at Chattogram hub. Collect from branch with this SMS and a photo ID. |
| 10063 | 0 | 5 | 0.350 | Dear customer, your Robi 5GB pack expires on 30 Nov. Dial *123*1# to renew. |
| 10079 | 0 | 3 | 0.330 | bKash: You have received Tk 2,000.00 from 01XXXXXXXXX. Fee Tk 0.00. Balance Tk 2,540.00. TrxID 8F2K1P |

## Scam predicted as legitimate

| id | true | confidence | text |
|---|---|---|---|
| 10056 | 5 | 0.511 | Book your dream apartment in Bashundhara, easy installment. Call 09611-XXXXXX |
| 10053 | 1 | 0.508 | mnbvcx Your bKash PIIIN will expire today Rpely witttth your currentP1N ad a new PIN to 💰 contiune service |
| 10064 | 4 | 0.452 | S: Hello, this is DHL. You have a package from Malaysia \|\|\| R: I didn't order anything \|\|\| S: It is registered under your name. Pay Tk 2,500 customs today or it will be returned |
| 10024 | 4 | 0.440 | Customs hold: your parcel from China needs Tk 1,850 duty. Pay via bKash to release. |
| 10070 | 1 | 0.429 | প্রিয় গ্রাহক, আপনার নগদ অ্যাকাউন্ট বন্ধ হয়ে যাবে। এখনই আপনার পিন নম্বর পাঠান। |
| 10055 | 5 | 0.407 | Pizza Hut combo: 2 large pizzas + drink for Tk 1,299 today only! |
| 10037 | 5 | 0.388 | S: Sir, hair fall problem? Our herbal oil is very popular \|\|\| S: 40% discount today only, order now |
| 10020 | 3 | 0.368 | ঘরে বসে প্রতিদিন ২০০০ টাকা আয় করুন। আজই হোয়াটসঅ্যাপে যোগ দিন। |
| 10035 | 1 | 0.367 | Amі agnt, bhulnumber-e taka pathaisi. Ferot dіn,PIN ta bolen. |
| 10009 | 1 | 0.360 | Daraz refund team: send card number, expiry and CVV to receive your Tk 1,500 refund. |
| 10015 | 5 | 0.356 | Learn English in 30 days! Admission going on. Call now. |
| 10062 | 3 | 0.354 | Congratulations, you are selected for Data Entry Officer at Unilever Bangladesh. Pay Tk 3,500 registration fee to confirm. |
| 10071 | 2 | 0.354 | Lucky draw winner! Reply with your name and address to receive a Toyota car. |
| 10038 | 5 | 0.347 | Robi:Special offer! zzqwpo 10GB for Tk 199, 7 days. Dal *123*199# |
| 10018 | 2 | 0.328 | S: Hello, you have been selected for our customer loyalty reward \|\|\| R: what reward? \|\|\| S: A free smartphone worth 45000 BDT. Just pay 500 delivery charge |
| 10082 | 5 | 0.317 | Aarong: E​nd of season sale up to 50% off Visit your qzxvbn ne4r357 OUTLET T&C apply |
| 10002 | 5 | 0.315 | Get 10% cashback on your next top-up with code EID10. |
| 10022 | 3 | 0.309 | S: Hi, I saw your CV on Bdjobs. We wkrjfejejrn have a work-from-home position \|\|\| R: what iste sаlаry? \|\|\| 🎁 S: 40K per month \|\|\| R: ok \|\|\| S: You need to deposit Tk2,000 assecurity first. xkcdpl |
| 10046 | 2 | 0.304 | অভিনন্দন! আপনি ১০,০০০ টাকা ক্যাশব্যাক জিতেছেন। পেতে নিচের লিংকে ক্লিক করুন। |
| 10060 | 5 | 0.304 | GP Mega Offer! Buy 1GB get 1GB free. Dial *121*3# |
| 10067 | 4 | 0.301 | Parcel atke ase. Delivery fee dite link e jan. |
| 10061 | 5 | 0.300 | ঈদ উপলক্ষে সব পণ্যে ২০% ছাড়! আজই কিনুন। |
| 10065 | 3 | 0.296 | Freelancing course: earn $500/month from home guaranteed. Enroll now with Tk 2,000. |
| 10008 | 5 | 0.292 | Daraz 11.11 sale starts tonight 12 AM. Vouchers up to Tk 1,000. |
| 10054 | 4 | 0.287 | Pathao Courier: Apnar parcel er address vul. Call korun 01XXXXXXXXX ekhoni. |
