# Retry: display only a preview (head) to avoid buffer limits, still save full CSV
import pandas as pd

# Reuse get_meta_df from previous cell if available; otherwise redefine quickly
def get_meta_df():
    records = []

    # STAI items
    qtext = {
        1: "1. I feel calm", 2: "2. I feel secure", 3: "3. I am tense", 4: "4. I feel strained",
        5: "5. I feel at ease", 6: "6. I feel upset", 7: "7. I am presently worrying over possible misfortunes",
        8: "8. I feel satisfied", 9: "9. I feel frightened", 10: "10. I feel comfortable",
        11: "11. I feel self-confident", 12: "12. I feel nervous", 13: "13. I am jittery",
        14: "14. I feel indecisive", 15: "15. I am relaxed", 16: "16. I feel content",
        17: "17. I am worried", 18: "18. I feel confused", 19: "19. I feel steady",
        20: "20. I feel pleasant", 21: "21. I feel pleasant", 22: "22. I feel nervous and restless",
        23: "23. I feel satistied with myself", 24: "24. I wish I could be as happy as others seem to be",
        25: "25. I feel like a failure", 26: "26. I feel rested", 27: "27. I am cool, calm, and collected",
        28: "28. I feel difficulties are piling up so that I cannot overcome them",
        29: "29. I worry too much over something that really doesn't matter", 30: "30. I am happy",
        31: "31. I have disturbing thoughts", 32: "32. I lack self-confidence", 33: "33. I feel secure",
        34: "34. I make decisions easily", 35: "35. I feel inadequate", 36: "36. I am content",
        37: "37. Some unimportant thought runs through my mind and bothers me",
        38: "38. I take disappointments so keenly that I can't put them out of my mind",
        39: "39. I am a steady person",
        40: "40. I get in a state of tension or turmoil as I think over my recent concerns and interests",
    }
    choices_1_20 = "1 = Not at all; 2 = Somewhat; 3 = Moderately; 4 = Very much"
    choices_21_40 = "1 = Almost never; 2 = Somewhat; 3 = Often; 4 = Almost always"

    for q in range(1, 41):
        label = qtext[q]
        section = "State items (1–20)" if q <= 20 else "Trait items (21–40)"
        choices = choices_1_20 if q <= 20 else choices_21_40
        for tp in range(0, 4):
            records.append({
                "instrument": "STAI",
                "section": section,
                "domain": "",
                "variable": f"stai_{str(q).zfill(2)}_tp{tp}",
                "label": label,
                "field_type": "num",
                "choices": choices,
                "timepoint": tp,
                "item_number": q
            })

    for tp in range(0, 4):
        records.append({
            "instrument": "STAI", "section": "Scores", "domain": "State",
            "variable": f"stai_state_score_tp{tp}",
            "label": "Self-Evaluation Questionnaire (STAI) State Score",
            "field_type": "num", "choices": "", "timepoint": tp, "item_number": ""
        })
    for tp in range(0, 4):
        records.append({
            "instrument": "STAI", "section": "Scores", "domain": "Trait",
            "variable": f"stai_trait_score_tp{tp}",
            "label": "Self-Evaluation Questionnaire (STAI) Trait Score",
            "field_type": "num", "choices": "", "timepoint": tp, "item_number": ""
        })
    for tp in range(0, 4):
        records.append({
            "instrument": "STAI", "section": "Admin", "domain": "",
            "variable": f"selfevaluation_complete_tp{tp}",
            "label": "Complete?", "field_type": "num", "choices": "",
            "timepoint": tp, "item_number": ""
        })

    # BDI-II
    bdi_items = {
        1: "1. Sadness", 2: "2. Pessimism", 3: "3. Past Failure", 4: "4. Loss of Pleasure",
        5: "5. Guilty Feelings", 6: "6. Punishment Feelings", 7: "7. Self-Dislike",
        8: "8. Self-Criticalness", 9: "9. Suicidal Thoughts or Wishes", 10: "10. Crying",
        11: "11. Agitation", 12: "12. Loss of Interest", 13: "13. Indecisiveness",
        14: "14. Worthlessness", 15: "15. Loss of Energy", 16: "16. Changes in Sleeping Pattern",
        17: "17. Irritability", 18: "18. Changes in Appetite", 19: "19. Concentration Difficulty",
        20: "20. Tiredness or Fatigue", 21: "21. Loss of Interest in Sex",
    }
    bdi_stems = [
        "", "sadness", "pessimism", "failure", "pleasure", "guilty", "punishment",
        "self_dislike", "self_criticalness", "suicidal_thoughts", "crying", "agitation",
        "interest", "indecisiveness", "worthlessness", "energy", "sleeping", "irritability",
        "appetite", "concentration", "tiredness", "sex"
    ]
    for i in range(1, 22):
        for tp in range(0, 4):
            records.append({
                "instrument": "BDI-II", "section": "Items", "domain": "",
                "variable": f"bdi_{bdi_stems[i]}_tp{tp}", "label": bdi_items[i],
                "field_type": "num", "choices": "", "timepoint": tp, "item_number": i
            })

    for tp in range(0, 4):
        records.append({
            "instrument": "BDI-II", "section": "Scores", "domain": "BDI Total (legacy)",
            "variable": f"bdi_total_tp{tp}", "label": "BDI Score:",
            "field_type": "num", "choices": "", "timepoint": tp, "item_number": ""
        })

    subscale_map = {
        "BDI_II_Cognitive": "BDI-II Cognitive Subscale",
        "BDI_II_Affective": "BDI-II Affective Subscale",
        "BDI_II_Somatic": "BDI-II Somatic Subscale",
        "BDI_II_Total": "BDI-II Total Score",
    }
    for base, friendly in subscale_map.items():
        for tp in range(0, 4):
            records.append({
                "instrument": "BDI-II", "section": "Scores",
                "domain": base.split("_")[-1].title().replace("Ii", "II"),
                "variable": f"{base}_tp{tp}", "label": friendly, "field_type": "num",
                "choices": "", "timepoint": tp, "item_number": ""
            })

    for tp in range(0, 4):
        records.append({
            "instrument": "BDI-II", "section": "Admin", "domain": "",
            "variable": f"bdiii_complete_tp{tp}", "label": "Complete?",
            "field_type": "num", "choices": "", "timepoint": tp, "item_number": ""
        })

    metadata_df = pd.DataFrame.from_records(
        records,
        columns=["instrument", "section", "domain", "variable", "label",
                 "field_type", "choices", "timepoint", "item_number"]
    ).sort_values(
        by=["instrument", "section", "timepoint", "item_number", "variable"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Add the requested demographics: age_v0_tp0..3 and sex_tp0..3
    demo_records = []
    for tp in range(0, 4):
        demo_records.append({
            "instrument": "Demographics",
            "section": "Core",
            "domain": "",
            "variable": f"age_v0_tp{tp}",
            "label": "Age",
            "field_type": "num",
            "choices": "",   # no choices for numeric age
            "timepoint": tp,
            "item_number": ""
        })
    for tp in range(0, 4):
        demo_records.append({
            "instrument": "Demographics",
            "section": "Core",
            "domain": "",
            "variable": f"sex_tp{tp}",
            "label": "Sex",
            "field_type": "num",
            "choices": "0 = Male; 1 = Female",
            "timepoint": tp,
            "item_number": ""
        })

    demo_df = pd.DataFrame.from_records(demo_records,
        columns=["instrument", "section", "domain", "variable", "label",
                "field_type", "choices", "timepoint", "item_number"]
    )

    updated_df = (
        pd.concat([metadata_df, demo_df], ignore_index=True)
        .sort_values(by=["instrument", "section", "timepoint", "item_number", "variable"],
                    kind="mergesort")
        .reset_index(drop=True)
    )
    metadata_df = updated_df

    return metadata_df

metadata_df = get_meta_df()

# Show a preview only
preview = metadata_df.head(100)
#display_dataframe_to_user("Preview: Instrument metadata (first 100 rows)", preview)

csv_path = "./data_generation_log/generated_table/metadata_stai_bdi_ii.csv"
metadata_df.to_csv(csv_path, index=False)
csv_path
