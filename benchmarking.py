# benchmarking.py (with GPT evaluation)
import requests
import time
import csv
from difflib import SequenceMatcher
from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

API_URL = "http://localhost:8000/api/v1/hackrx/run"
BLOB_URL = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))

TEST_SET_EASY = [
    {
      "question": "What is the official name of the insurance company?",
      "expected": "The official name of the company is CHOLAMANDALAM MS GENERAL INSURANCE COMPANY LIMITED[cite: 422, 431, 459, 493, 525, 553, 584, 609, 638]."
    },
    {
      "question": "What is the entry age for a member to be insured under this policy?",
      "expected": "The entry age for a member should be between 3 months to 90 years (completed age)[cite: 633]."
    },
    {
      "question": "What is the definition of a 'Family' in this policy?",
      "expected": "Family means the Insured, his/her lawful spouse and maximum of any two dependent children up to the age of 25 years, parents, parents-in laws, siblings, children's in laws, partner[cite: 546]."
    },
    {
      "question": "What is the definition of 'Accident / Accidental'?",
      "expected": "Accident / Accidental mean a sudden, unforeseen and involuntary event caused by external, visible and violent means[cite: 448]."
    },
    {
      "question": "What is the minimum period of hospitalization for 'In-patient Care'?",
      "expected": "Hospitalisation means admission in a Hospital for a minimum period of 24 consecutive 'In-patient Care' hours except for specified procedures/treatments, where such admission could be for a period of less than 24consecutive hours[cite: 475]."
    },
    {
      "question": "What is considered a 'Chronic condition'?",
      "expected": "A Chronic condition is defined as a disease, illness, or injury that has one or more of the following characteristics: it needs ongoing or long-term monitoring through consultations, examinations, check-ups, and/or tests; it needs ongoing or long-term control or relief of symptoms; it requires rehabilitation for the patient or for the patient to be specially trained to cope with it; it continues indefinitely; it recurs or is likely to recur[cite: 478]."
    },
    {
      "question": "What does 'Pre-existing Disease' mean?",
      "expected": "Pre-existing Disease means any condition, ailment, injury or disease that is/are diagnosed by a physician within 48 months prior to the effective date of the policy issued by the Insurer or its reinstatement or for which medical advice or treatment was recommended by, or received from, a physician within 48 months prior to the effective date of the policy issued by the insurer or its reinstatement[cite: 510, 511, 512]."
    },
    {
      "question": "What does the policy not cover under 'Contents'?",
      "expected": "The policy does not include deeds, bonds, bills of exchange, promissory notes, cheques, traveller's cheques, and securities for money, documents of any kind, cash, and currency notes[cite: 542]."
    },
    {
      "question": "What is the definition of 'Hijack'?",
      "expected": "Hijack means any unlawful seizure or exercise of control, by force or violence or threat of force or violence and with wrongful intent, of the Common Carrier in which the Insured Person is travelling[cite: 563]."
    },
    {
      "question": "What is the definition of 'Permanent Total Disablement'?",
      "expected": "Permanent Total Disablement' means a bodily injury caused by accidental, external, violent and visible means, which as a direct consequence thereof totally disables and prevents the insured from attending to any business or occupation of any and every kind or if he/she has no business or occupation, from attending to his/her usual and normal duties that last for a continuous period of twelve calendar months from the date of the accident, with no hopes of improvement at the end of that period[cite: 596]."
    },
    {
      "question": "What documents are required for a claim under 'Accidental Death'?",
      "expected": "For an Accidental Death claim, the required documents include a Police report in original (if the accident was in a public place), a Death Certificate, a Post Mortem Report (in case of death), a detailed sequence of events, and medical records detailing the accident and injury[cite: 355]."
    },
    {
      "question": "What is the maximum liability of the company for 'Personal Accident Covers'?",
      "expected": "The Sum Insured shall be the maximum liability of the Company under this benefit[cite: 14]."
    },
    {
      "question": "What is the policy regarding fraudulent claims?",
      "expected": "If any claim is found to be fraudulent, all benefits under the policy and the premium paid shall be forfeited. Any amount already paid must be repaid by the recipients[cite: 69, 70]."
    },
    {
      "question": "What is the definition of 'Trip'?",
      "expected": "'Trip' means a journey out of usual place of residence in India and back, the details of which are specified in the Policy Schedule/Certificate. It includes Business and Leisure trips unless specified otherwise[cite: 624, 625]."
    },
    {
      "question": "What is the maximum duration for a trip, including extensions?",
      "expected": "The maximum trip duration (including the extension) shall not exceed 365 days in total[cite: 54]."
    }
]

TEST_SET_MEDIUM = [
    {
        "question": "If an insured person is hospitalized for a pre-existing disease within 24 months of policy inception, will the claim be accepted?",
        "expected": "No, pre-existing diseases are only covered after 36 months of continuous coverage."
    },
    {
        "question": "Is cosmetic surgery covered if required due to an accident?",
        "expected": "Yes, cosmetic surgery is covered if it is necessary due to an accidental injury and not for aesthetic reasons."
    },
    {
        "question": "Can an insured person claim for organ donor expenses if the organ is not donated to an insured family member?",
        "expected": "No, organ donor expenses are covered only if the organ is donated to an Insured Person under the policy."
    },
    {
        "question": "If a claim is made during the grace period, will it be accepted?",
        "expected": "Claims made during the grace period will not be admissible unless the premium is paid within the grace period and the policy is renewed without a break."
    },
    {
        "question": "If the insured is hospitalized due to a condition excluded under the policy, will they receive hospital cash benefit?",
        "expected": "No, hospital cash benefit is not available for conditions listed under policy exclusions."
    },
    {
        "question": "If a person is hospitalized for 4 days, how much hospital cash benefit can they claim?",
        "expected": "They can claim INR 500 per day after the first 3 days, so only INR 500 is payable."
    },
    {
        "question": "Will the policy cover cataract surgery for both eyes in a single policy year?",
        "expected": "The policy will cover cataract surgery for both eyes but the total benefit is limited to INR 60,000 or 15% of the Sum Insured in a policy year."
    },
    {
        "question": "Can expenses for infertility treatment be claimed in the first policy year?",
        "expected": "No, infertility treatment is covered only after a waiting period of two years."
    },
    {
        "question": "If a child above 18 is studying but financially dependent, are they eligible for coverage as a child?",
        "expected": "Yes, dependent children are covered up to 25 years of age if they are financially dependent and unmarried."
    },
    {
        "question": "Are OPD dental expenses covered under the policy for routine checkups?",
        "expected": "No, dental treatment is covered only if necessitated by an injury."
    },
    {
        "question": "Will the policy cover expenses for treatment received in an AYUSH hospital for a listed illness?",
        "expected": "Yes, provided the AYUSH hospital is government recognized and the treatment is for a covered illness."
    },
    {
        "question": "Can an insured person claim both maternity and newborn baby cover in the same policy year?",
        "expected": "Yes, provided the maternity benefit waiting period is completed and the newborn is added as an insured within 90 days of birth."
    },
    {
        "question": "Will ambulance charges be reimbursed if hospitalization is less than 24 hours?",
        "expected": "Ambulance charges are reimbursed only if hospitalization is admissible under the policy."
    },
    {
        "question": "If an insured person is diagnosed with HIV during the policy period, are all related expenses covered?",
        "expected": "Yes, treatment for HIV/AIDS is covered as per policy terms, subject to sub-limits."
    },
    {
        "question": "If two claims are made in the same policy year, will both be eligible for No Claim Discount?",
        "expected": "No, No Claim Discount is only available if no claim is made in the policy year."
    },
    {
        "question": "If the insured person opts for a higher room rent than allowed, will the entire claim be paid?",
        "expected": "No, if room rent exceeds the limit, proportionate deductions apply as per policy terms."
    },
    {
        "question": "Is there any coverage for complications arising from maternity?",
        "expected": "Yes, complications arising from maternity are covered, subject to waiting period and policy limits."
    },
    {
        "question": "If an insured person receives alternative treatment at a private AYUSH hospital, will the expenses be reimbursed?",
        "expected": "Yes, if the hospital is government recognized and the treatment is inpatient."
    },
    {
        "question": "Are expenses for organ transplantation surgery covered for the donor and recipient under the same policy?",
        "expected": "Yes, provided both donor and recipient are insured persons under the policy."
    },
    {
        "question": "If the insured person undergoes robotic surgery for a covered condition, what is the maximum claimable amount?",
        "expected": "Up to 25% of the Sum Insured is claimable for robotic surgery under modern treatments."
    }
]

TEST_SET_HARD = [
    {
      "question": "A family of three (insured, spouse, and one child) has a policy. The child, who is 26 years old, is diagnosed with a 'chronic condition' which was first detected 40 months before the policy's effective date. The child is hospitalized for a minor procedure that requires less than 24 consecutive hours of 'In-patient Care'. Will the policy cover this claim? Justify your answer using the definitions of 'Family', 'Dependent Children', 'Chronic condition', 'Pre-existing Disease', and 'In-patient Care'.",
      "expected": "The policy would not cover this claim. First, the definition of a 'Family' includes dependent children up to the age of 25. Since the child is 26, they are not a dependent child under the policy's definition. Second, even if the child were a dependent, the 'chronic condition' would be classified as a 'Pre-existing Disease' because it was diagnosed within 48 months of the policy's effective date. Therefore, the treatment would be excluded from coverage."
    },
    {
      "question": "A policyholder suffers an accident that leads to 'Permanent Total Disablement'. The injury occurred during a trip outside their usual place of residence. The policyholder makes a claim for both the medical expenses and the disablement benefit. What are the key definitions and conditions the company would examine to validate the claim for both benefits, and what would be the maximum liability for the disablement benefit?",
      "expected": "The company would examine the definitions of 'Accident/Accidental', 'Permanent Total Disablement', and 'Trip'. They would need to confirm that the injury was caused by an 'Accident' and that it resulted in a disablement that prevents the insured from working for a continuous period of twelve months with no hope of improvement, meeting the definition of 'Permanent Total Disablement'. The 'Trip' definition would be reviewed to ensure the journey was covered. The maximum liability for the 'Permanent Total Disablement' benefit would be the Sum Insured."
    },
    {
      "question": "A 70-year-old insured person is diagnosed with a 'critical illness' (as per the policy's list) 50 months after the policy's effective date. The insured is admitted for 'In-patient Care' for 28 hours. The illness requires 'ongoing or long-term monitoring'. Can this be considered a 'chronic condition', and what is the policy's stance on this claim considering the age and timing of the diagnosis?",
      "expected": "Yes, this can be considered a 'chronic condition' as it requires 'ongoing or long-term monitoring'. The policy's entry age is up to 90 years, so the insured's age is within the limit. Since the illness was diagnosed 50 months after the policy's effective date, it would not be a 'Pre-existing Disease'. The hospitalization for 28 hours meets the 'In-patient Care' definition. Therefore, the claim would likely be covered, assuming all other policy conditions are met."
    },
    {
      "question": "A business trip lasts for 350 days. The insured suffers a bodily injury due to an accidental event. A fraudulent claim is filed by a family member to get benefits. The company investigates and finds the claim to be fraudulent. What is the consequence for the policy and the fraudulent claimant? What is the maximum duration for a trip according to the policy?",
      "expected": "The maximum duration for a trip (including extensions) is 365 days, so a 350-day trip is within the limit. If a fraudulent claim is made, all benefits under the policy are forfeited, and the premium paid is also forfeited. Any amount already paid must be repaid by the recipients. The consequence is severe, as per the 'Fraudulent Claims' clause in the policy."
    },
    {
      "question": "A member is admitted to a hospital for a series of 'specified procedures/treatments' that individually take less than 24 hours but, combined, result in a total hospitalization of 48 hours. The member is over 90 years of age. Would this be considered 'In-patient Care'? What is the age limit for a member to be insured?",
      "expected": "The policy defines 'In-patient Care' as admission for a minimum of 24 consecutive hours, with an exception for specified procedures/treatments where the admission could be for less than 24 hours. In this case, the total hospitalization is 48 hours, which meets the 24-hour criteria. However, the entry age for a member is up to 90 years. Since the member is over 90, they would not be eligible to be insured under this policy."
    },
    {
      "question": "A claim is filed for a 'Permanent Total Disablement'. The claimant provides all required documentation, including police reports, but the company's internal investigation reveals that the details provided in the 'detailed sequence of events' are inconsistent with the 'medical records detailing the accident and injury'. What would be the likely outcome, and what policy clause would be used to justify the decision?",
      "expected": "The claim would likely be denied. The company's 'Claim Procedure and Requirements' section requires documentation to be consistent. The inconsistency between the 'detailed sequence of events' and the 'medical records' would raise a red flag. Furthermore, if the company finds the claim to be fraudulent or based on misrepresentation, the 'Fraudulent Claims' clause would apply, leading to the forfeiture of all policy benefits and premium."
    },
    {
      "question": "An insured's spouse, aged 50, has a 'Pre-existing Disease' that was diagnosed 45 months prior to the policy's effective date. The spouse is hospitalized for a different illness that is not a pre-existing condition, and the hospitalization lasts for 30 hours. Is the spouse considered part of the 'Family', and would the new hospitalization be covered?",
      "expected": "Yes, the spouse is part of the 'Family' as per the definition, which includes the 'lawful spouse'. The hospitalization for 30 hours meets the 'In-patient Care' definition. The fact that the spouse has a 'Pre-existing Disease' is irrelevant to the new claim, as the hospitalization is for a different illness that is not a pre-existing condition. Therefore, the claim for the new illness would be covered."
    },
    {
      "question": "A policyholder's parent, who is 85 years old, is insured under the policy as part of the 'Family'. The parent suffers a 'Chronic condition' that was first diagnosed 5 years before the policy's effective date. The parent is admitted for 'In-patient Care' for this condition. Is the parent eligible for coverage, and would this claim be paid? Why or why not?",
      "expected": "The parent is eligible for coverage, as they are part of the 'Family' and their age (85) is within the entry age limit (up to 90 years). The claim for the 'Chronic condition' would be covered because it was diagnosed 5 years (60 months) before the policy's effective date, which is outside the 48-month window for a 'Pre-existing Disease'. Therefore, the condition is not considered pre-existing and is eligible for coverage."
    },
    {
      "question": "An insured person is traveling and their common carrier is 'hijacked'. During the event, they suffer a bodily injury that results in a disablement that lasts for 10 months, after which they fully recover. Can they claim for 'Permanent Total Disablement', and why?",
      "expected": "No, they cannot claim for 'Permanent Total Disablement'. The definition requires the disablement to last for a 'continuous period of twelve calendar months' with 'no hopes of improvement at the end of that period'. Since the disablement lasted for only 10 months and the insured fully recovered, it does not meet the criteria for a 'Permanent Total Disablement' claim."
    },
    {
      "question": "A policyholder has a policy with a grace period of thirty days for premium payment. The due date is December 1st. The policyholder pays the premium on December 28th. A claim is filed for a hospitalization that occurred on December 15th. Is the claim valid, and what is the policy's stance on this?",
      "expected": "Yes, the claim is valid. The policy provides a grace period of thirty days for premium payment after the due date. The payment on December 28th is within this grace period. The policy remains in force during the grace period, so the claim for the hospitalization on December 15th would be processed."
    },
    {
      "question": "A person has a policy for which the premium is due on January 1st. A claim is made for a medical expense that occurred on January 30th. The policyholder renews the policy by paying the premium on January 31st. The company finds that the claim is fraudulent. What is the consequence of the fraudulent claim and the late premium payment? How does the 'grace period' affect this scenario?",
      "expected": "The grace period for premium payment is thirty days, so the premium payment on January 31st is within the grace period and the policy is considered renewed. However, if the claim is found to be fraudulent, the 'Fraudulent Claims' clause takes precedence. All benefits under the policy and the premium paid would be forfeited, and any amount already paid would need to be repaid. The fraudulent claim nullifies any benefit from the grace period."
    },
    {
      "question": "The policy schedule specifies that a 'Trip' includes 'Business and Leisure trips unless specified otherwise'. If a policyholder is on a pilgrimage, which is not explicitly mentioned, would this be considered a covered 'Trip'? Explain your reasoning.",
      "expected": "Yes, it would likely be considered a covered 'Trip'. The policy states that a trip includes 'Business and Leisure trips unless specified otherwise'. A pilgrimage, while not a typical 'business' or 'leisure' trip, is a journey out of the usual place of residence and back, which is the core definition of a 'Trip'. Unless the policy explicitly excludes pilgrimages or journeys of a religious nature, it would be covered."
    },
    {
      "question": "A 'Family' policy covers the insured, spouse, and two children. The insured's parents also live with them and are dependent on the insured. The insured adds the parents to the policy. Are the parents covered, and under which definition?",
      "expected": "Yes, the parents are covered. The definition of 'Family' explicitly includes 'parents' in addition to the insured, lawful spouse, and a maximum of any two dependent children. So, the addition of the parents to the policy would be in line with the policy's definitions."
    },
    {
      "question": "A policyholder's claim for a 'critical illness' is initially approved, and a payment is made. Later, the company finds out that the medical records submitted were fraudulent and that the 'illness' was a 'Pre-existing Disease' that was diagnosed 40 months prior to the policy's effective date. What action will the company take, and what is the basis for it?",
      "expected": "The company will take action based on the 'Fraudulent Claims' clause. Since the medical records were fraudulent and the claim was for a 'Pre-existing Disease' (diagnosed within the 48-month window), the company will forfeit all benefits and the premium paid. The amount already paid to the policyholder must be repaid to the company."
    },
    {
      "question": "A policyholder suffers an injury that results in a temporary disablement for 11 months. The disablement is a direct consequence of an accident. The policyholder claims for 'Permanent Total Disablement'. The company denies the claim. What is the basis for the denial, and what is the key difference between temporary and permanent total disablement in this context?",
      "expected": "The basis for the denial is that the disablement did not last for a 'continuous period of twelve calendar months'. The key difference is the duration of the disablement. 'Permanent Total Disablement' specifically requires a minimum continuous period of twelve months with no hope of improvement. A disablement of 11 months, even if a result of an accident, does not meet this specific criterion."
    },
    {
      "question": "A policyholder has 'Contents' insurance. The insured is robbed, and several valuable documents, including deeds and promissory notes, are stolen. The insured files a claim for the value of these documents. Will the claim be paid, and why?",
      "expected": "No, the claim will not be paid. The 'Contents' definition in the policy explicitly excludes 'deeds, bonds, bills of exchange, promissory notes, cheques, traveller's cheques, and securities for money, documents of any kind, cash, and currency notes'. Therefore, the stolen documents are not covered under the policy."
    },
    {
      "question": "A policyholder is traveling on a flight that is unlawfully seized by a third party. The policyholder suffers a minor injury that does not require hospitalization. The policyholder files a claim under the 'Hijack' benefit. Will this claim be paid, and what are the specific criteria for a 'Hijack' claim under this policy?",
      "expected": "The claim would likely not be paid under the 'Personal Accident Covers' section, as the 'Hijack' definition is for the 'unlawful seizure or exercise of control' of the common carrier, which is a condition of the event itself. However, the policyholder would need to meet the criteria for a 'Personal Accident Covers' claim, which is typically for bodily injury or death. The policy states that the sum insured is the maximum liability. However, since the injury is minor and does not meet the specified criteria (e.g., 'Permanent Total Disablement' or 'Accidental Death'), the claim may not be covered."
    },
    {
      "question": "A policyholder is on a 'Trip' that has been extended from 350 days to 380 days. During the extension, they are involved in a traffic accident. Is the accident covered under the policy, and what is the maximum duration for a trip?",
      "expected": "The accident would not be covered. The policy states that the maximum trip duration, including extensions, is 365 days. Since the trip was extended to 380 days, it exceeds the policy's limit, and any incident that occurs during the period after the 365th day would not be covered."
    },
    {
      "question": "A person has a 'Pre-existing Disease' that was diagnosed 47 months prior to the policy's effective date. They are hospitalized for a minimum of 24 consecutive hours for a treatment related to this disease. The claim is submitted with a complete set of documentation. What is the likely outcome of the claim?",
      "expected": "The claim would be denied. The definition of a 'Pre-existing Disease' is any condition diagnosed within 48 months prior to the effective date of the policy. Since the disease was diagnosed 47 months before the policy, it falls within this window and is excluded from coverage."
    },
    {
      "question": "The policy defines a 'Family' to include the insured's siblings. The insured's brother, who is 30 years old and lives separately, is added to the policy. The brother is hospitalized for a heart condition that was diagnosed 4 years prior to the policy's effective date. Is the brother a part of the family as per the policy and would the heart condition be covered?",
      "expected": "Yes, the brother is part of the 'Family' as the definition includes 'siblings'. The heart condition would be covered because it was diagnosed 4 years (48 months) prior to the policy's effective date. This is not a 'Pre-existing Disease' as per the policy's definition, which specifies 'within 48 months'. Since it's exactly 48 months or more, it is not considered pre-existing and is therefore covered."
    }
]

def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def gpt_judgment(question, expected, predicted):
    prompt = f"""
You are an expert evaluator. Given the original question, the expected expected, and the model's predicted expected,
rate how well the predicted answer aligns with the expected answer on a scale of 0.0 to 1.0.

Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Justify your score and then provide the final numeric score.
Respond in this format:
{{
  "score": float,
  "explanation": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256
    )


    print(response.choices[0].message.content.strip())
    text = response.choices[0].message.content.strip()
    json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
    return json.loads(json_block)
    # try:
    # except:
    #     return {"score": 0.0, "explanation": "Invalid JSON returned."}


def run_benchmark():
    results = []
    questions = [q["question"] for q in TEST_SET_HARD]
    payload = {
        "documents": BLOB_URL,
        "questions": questions
    }

    start_time = time.time()
    response = requests.post(API_URL, json=payload)
    elapsed = time.time() - start_time

    response.raise_for_status()
    predictions = response.json().get("answers", [])

    for i, pred in enumerate(predictions):
        gt = TEST_SET_HARD[i]["expected"]
        predicted = pred["answer"]
        reasoning = pred.get("reasoning", "")
        clauses = ", ".join(pred.get("clauses", []))
        score = fuzzy_match(predicted, gt)

        gpt_eval = gpt_judgment(pred["question"], gt, predicted)

        results.append({
            "Question": pred["question"],
            "Predicted Answer": predicted,
            "Expected": gt,
            "Fuzzy Score": round(score, 3),
            "GPT Score": gpt_eval.get("score", 0.0),
            "GPT Reason": gpt_eval.get("explanation", "N/A"),
            "Reasoning": reasoning,
            "Clauses": clauses,
            "Latency (s)": round(elapsed, 2)
        })

    with open("rag_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # === Add this block ===
    avg_fuzzy = sum(r["Fuzzy Score"] for r in results) / len(results)
    avg_gpt = sum(r["GPT Score"] for r in results) / len(results)

    print(f"\n📊 Average Fuzzy Score: {avg_fuzzy:.3f}")
    print(f"🤖 Average GPT Score:   {avg_gpt:.3f}")

    print("✅ Benchmark complete. Results saved to rag_benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
