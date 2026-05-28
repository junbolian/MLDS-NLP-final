# Error Analysis Cases

This file is the human-readable companion to [`notebooks/06_error_analysis.ipynb`](../notebooks/06_error_analysis.ipynb).
The per-case prediction CSVs were exported from the saved §3 model artifacts with:

```bash
python -m app.export_predictions --all --split test
```

Important scope note: the current §3 classifiers are trained and evaluated on `long_ref`, the dataset's human-written long summary. That means the four classification cases below quote the `long_ref` text that actually drove the saved model predictions. Only the summarization case (`CJ-AL-0020`) uses `long_pred` from [`abstractive_summaries.csv`](abstractive_summaries.csv).

## 1. Model disagreement

- `case_id`: `IM-CA-0025`
- `text analyzed (long_ref excerpt)`: "This was a class action ... on behalf of all legal hourly-paid agricultural workers ... Plaintiff alleged that employees of Fruit Patch, Inc. engaged in a massive scheme to hire undocumented immigrants ... Plaintiff alleges that defendants' scheme violated the Racketeer Influenced and Corrupt Organizations Act (RICO) ... and the Immigration and Nationality Act ..."
- `true labels`: class action = `Yes`; case type = `Immigration & Education`
- `predicted labels`:
  - class action: NB `Yes (0.793)`, LR `Yes (0.623)`, Bi-LSTM `Yes (0.999)`, BERT `Yes (0.994)`
  - case type: NB `Immigration & Education (0.361)`, LR `Criminal Justice (0.348)`, Bi-LSTM `Civil Rights & Equality (0.993)`, BERT `Immigration & Education (0.940)`
- `analysis`: This is the cleanest all-model disagreement example because the legal story sits across multiple semantic frames at once: labor exploitation, undocumented immigration, and RICO-style enterprise language. The sparse models react to different cue families, while BERT and, interestingly, NB recover the gold label. It is a good reminder that the grouped case-type task is not just about keywords; it is about which legal frame dominates when several plausible ones coexist.

## 2. Summary hallucination or unsupported detail

- `case_id`: `CJ-AL-0020`
- `reference evidence (long_ref excerpt)`: "The plaintiff sued the City of Montgomery and the Honorable Milton J. Westry under 42 U.S.C. § 1983 ... requested the court quash the Municipal Court order requiring the petitioner to serve an imprisonment term of 54 days ... claimed that the Municipal Court order violated Sixth Amendment, due process, and equal protection ..."
- `generated long summary (long_pred excerpt)`: "Harriet Cleveland was ordered to serve 31 days in jail because of her inability to pay fines and fees on multiple traffic tickets ... is facing the imminent loss of her home ... You have a right to appeal the decision of the Montgomery Municipal Court ... If you still believe the 'Balance Due' amount is incorrect ..."
- `evaluation`: ROUGE-1 `0.162`, ROUGE-L `0.092`, BERTScore F1 `0.736`
- `analysis`: The generated summary keeps the broad debtors'-prison theme but drifts into unsupported or poorly grounded details such as the "imminent loss of her home" claim and appeal-form boilerplate. It also drops much of the procedural/legal framing that matters in the reference summary, including the named defendants and the constitutional claims. This is exactly the kind of divergence Jianong should flag: the summary is not random gibberish, but it shifts emphasis in ways that could mislead downstream interpretation.

## 3. High confidence but wrong

- `case_id`: `DR-PA-0008`
- `text analyzed (long_ref excerpt)`: "The lawsuit was brought under the Rehabilitation Act, the Americans with Disabilities Act, and 42 U.S.C. § 1983 ... The plaintiff alleged that the PDOC systematically denied medical care to persons with severe eye conditions, including severe cataracts ... the DOC allegedly had an administrative policy colloquially known as the 'One Good Eye' policy ..."
- `true labels`: class action = `Yes`; case type = `Healthcare & Disability`
- `predicted labels`:
  - case type: NB `Criminal Justice (0.968)`, LR `Criminal Justice (0.784)`, Bi-LSTM `Criminal Justice (0.694)`, BERT `Criminal Justice (0.896)`
- `analysis`: This is the strongest overconfidence example because all four classifiers miss the same label for the same reason: the prison setting dominates the surface form. The gold label, however, is driven by the substantive disability-rights and medical-care claims, not by the incarceration venue itself. The case shows why rare classes are hard here: if the model gives too much weight to prison words and too little to ADA / Rehabilitation Act language, it snaps to the much more common Criminal Justice bucket.

## 4. Low confidence but correct

- `case_id`: `NS-DC-0123`
- `text analyzed (long_ref excerpt)`: "The American Civil Liberties Union Foundation filed this lawsuit ... on behalf of an American citizen being detained by the United States military in Iraq ... under the federal habeas corpus statute ... The Defense Department asserted it was detaining the plaintiff because he was allegedly fighting for ISIS ..."
- `true labels`: class action = `No`; case type = `Speech & Voting`
- `predicted labels`:
  - class action: NB `No (0.766)`, LR `No (0.608)`, Bi-LSTM `No (0.995)`, BERT `No (0.973)`
  - case type: NB `Criminal Justice (0.416)`, LR `Criminal Justice (0.411)`, Bi-LSTM `Speech & Voting (0.803)`, BERT `Speech & Voting (0.400)`
- `analysis`: BERT gets the case type right here, but only barely, and that low confidence is informative rather than bad. The case mixes detention, military custody, terrorism allegations, and habeas procedure, so it genuinely straddles Criminal Justice and Speech/Voting-style constitutional/executive-power litigation. The low confidence is therefore a faithful signal that this example sits near the decision boundary.

## 5. Rare-class failure or recovery

- `case_id`: `DR-CA-0033`
- `text analyzed (long_ref excerpt)`: "A paraplegic individual who uses a wheelchair sued Chipotle ... alleging that two of the defendant's restaurants did not provide full and equal access to customers in wheelchairs ... in violation of the Americans With Disabilities Act (ADA), the Rehabilitation Act, and California civil-rights statutes ..."
- `true labels`: class action = `No`; case type = `Healthcare & Disability`
- `predicted labels`:
  - class action: NB `Yes (0.830)`, LR `Yes (0.598)`, Bi-LSTM `No (0.997)`, BERT `No (0.965)`
  - case type: NB `Healthcare & Disability (0.319)`, LR `Criminal Justice (0.367)`, Bi-LSTM `Civil Rights & Equality (0.970)`, BERT `Healthcare & Disability (0.749)`
- `analysis`: This case is the best rare-class recovery example because the text is unambiguously disability-centered, yet only NB and BERT recover the gold label. LR collapses to Criminal Justice and Bi-LSTM treats it like a broader equality/discrimination case, which shows how fragile the tail class is. Across the 16 `Healthcare & Disability` test cases, BERT gets 10 correct while LR gets 0, so this example is not an isolated win; it reflects the overall long-tail pattern.
