# Author: Harsh Kohli
# Date Created: 03-09-2024

BASE_MODEL = "google/flan-t5-large"
DATASETS_DIR = "resources/flan_datasets"
MODELS_DIR = "resources/lorahub"
DIFFICULTY_DIR = "resources/difficulty_scores"
MAPPINGS_FILE = "resources/all_mappings.json"
ABILITIES_FILE = "resources/abilities.json"

LORA_MODULE_NAMES = [
    "lorahub/flan_t5_large-qasc_qa_with_separated_facts_3",
    "lorahub/flan_t5_large-ag_news_subset",
    "lorahub/flan_t5_large-web_questions_whats_the_answer",
    "lorahub/flan_t5_large-wiki_hop_original_choose_best_object_affirmative_1",
    "lorahub/flan_t5_large-quoref_What_Is_The_Answer",
    "lorahub/flan_t5_large-qasc_is_correct_1",
    "lorahub/flan_t5_large-ropes_given_background_situation",
    "lorahub/flan_t5_large-duorc_SelfRC_title_generation",
    "lorahub/flan_t5_large-wiki_hop_original_choose_best_object_affirmative_3",
    "lorahub/flan_t5_large-wiki_hop_original_generate_subject",
    "lorahub/flan_t5_large-coqa",
    "lorahub/flan_t5_large-adversarial_qa_droberta_question_context_answer",
    "lorahub/flan_t5_large-amazon_polarity_flattering_or_not",
    "lorahub/flan_t5_large-quarel_choose_between",
    "lorahub/flan_t5_large-adversarial_qa_dbidaf_based_on",
    "lorahub/flan_t5_large-adversarial_qa_dbert_answer_the_following_q",
    "lorahub/flan_t5_large-dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to",
    "lorahub/flan_t5_large-wiki_hop_original_choose_best_object_interrogative_1",
    "lorahub/flan_t5_large-trec",
    "lorahub/flan_t5_large-race_high_Write_a_multi_choice_question_options_given_",
    "lorahub/flan_t5_large-social_i_qa_Show_choices_and_generate_answer",
    "lorahub/flan_t5_large-app_reviews_categorize_rating_using_review",
    "lorahub/flan_t5_large-wiki_hop_original_generate_subject_and_object",
    "lorahub/flan_t5_large-true_case",
    "lorahub/flan_t5_large-wiki_qa_Topic_Prediction_Answer_Only",
    "lorahub/flan_t5_large-quartz_given_the_fact_answer_the_q",
    "lorahub/flan_t5_large-quail_context_question_description_answer_text",
    "lorahub/flan_t5_large-dbpedia_14_given_a_choice_of_categories_",
    "lorahub/flan_t5_large-dream_baseline",
    "lorahub/flan_t5_large-wiki_qa_Is_This_True_",
    "lorahub/flan_t5_large-glue_wnli",
    "lorahub/flan_t5_large-adversarial_qa_dbert_based_on",
    "lorahub/flan_t5_large-quoref_Read_And_Extract_",
    "lorahub/flan_t5_large-amazon_polarity_User_recommend_this_product",
    "lorahub/flan_t5_large-wiqa_what_is_the_final_step_of_the_following_process",
    "lorahub/flan_t5_large-ropes_plain_no_background",
    "lorahub/flan_t5_large-wiki_hop_original_choose_best_object_affirmative_2",
    "lorahub/flan_t5_large-race_middle_Select_the_best_answer_generate_span_",
    "lorahub/flan_t5_large-quoref_Answer_Question_Given_Context",
    "lorahub/flan_t5_large-wmt16_translate_tr-en",
    "lorahub/flan_t5_large-quoref_Found_Context_Online",
    "lorahub/flan_t5_large-wiki_qa_Decide_good_answer",
    "lorahub/flan_t5_large-para_crawl_enes",
    "lorahub/flan_t5_large-race_middle_Taking_a_test",
    "lorahub/flan_t5_large-ropes_background_new_situation_answer",
    "lorahub/flan_t5_large-fix_punct",
    "lorahub/flan_t5_large-super_glue_rte",
    "lorahub/flan_t5_large-ropes_background_situation_middle",
    "lorahub/flan_t5_large-race_high_Taking_a_test",
    "lorahub/flan_t5_large-wiki_bio_who",
    "lorahub/flan_t5_large-quartz_paragraph_question_plain_concat",
    "lorahub/flan_t5_large-ropes_plain_background_situation",
    "lorahub/flan_t5_large-quoref_Given_Context_Answer_Question",
    "lorahub/flan_t5_large-adversarial_qa_dbidaf_question_context_answer",
    "lorahub/flan_t5_large-wmt16_translate_ro-en",
    "lorahub/flan_t5_large-adversarial_qa_dbert_question_context_answer",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_question_answering",
    "lorahub/flan_t5_large-race_high_Is_this_the_right_answer",
    "lorahub/flan_t5_large-sciq_Direct_Question",
    "lorahub/flan_t5_large-super_glue_wsc.fixed",
    "lorahub/flan_t5_large-super_glue_wic",
    "lorahub/flan_t5_large-quoref_Answer_Friend_Question",
    "lorahub/flan_t5_large-imdb_reviews_plain_text",
    "lorahub/flan_t5_large-race_middle_Select_the_best_answer",
    "lorahub/flan_t5_large-quail_context_question_answer_description_id",
    "lorahub/flan_t5_large-wiki_qa_found_on_google",
    "lorahub/flan_t5_large-glue_sst2",
    "lorahub/flan_t5_large-quail_context_description_question_answer_id",
    "lorahub/flan_t5_large-super_glue_cb",
    "lorahub/flan_t5_large-ropes_prompt_bottom_no_hint",
    "lorahub/flan_t5_large-anli_r1",
    "lorahub/flan_t5_large-ropes_read_background_situation",
    "lorahub/flan_t5_large-qasc_qa_with_separated_facts_2",
    "lorahub/flan_t5_large-quarel_heres_a_story",
    "lorahub/flan_t5_large-social_i_qa_Generate_the_question_from_the_answer",
    "lorahub/flan_t5_large-sciq_Multiple_Choice_Closed_Book_",
    "lorahub/flan_t5_large-math_dataset_algebra__linear_1d",
    "lorahub/flan_t5_large-yelp_polarity_reviews",
    "lorahub/flan_t5_large-adversarial_qa_droberta_tell_what_it_is",
    "lorahub/flan_t5_large-wiqa_what_might_be_the_last_step_of_the_process",
    "lorahub/flan_t5_large-adversarial_qa_dbidaf_answer_the_following_q",
    "lorahub/flan_t5_large-quoref_Guess_Answer",
    "lorahub/flan_t5_large-amazon_polarity_convey_negative_or_positive_sentiment",
    "lorahub/flan_t5_large-wiki_qa_Topic_Prediction_Question_Only",
    "lorahub/flan_t5_large-ropes_new_situation_background_answer",
    "lorahub/flan_t5_large-web_questions_potential_correct_answer",
    "lorahub/flan_t5_large-qasc_is_correct_2",
    "lorahub/flan_t5_large-quoref_Find_Answer",
    "lorahub/flan_t5_large-app_reviews_convert_to_rating",
    "lorahub/flan_t5_large-quail_description_context_question_answer_text",
    "lorahub/flan_t5_large-qasc_qa_with_separated_facts_4",
    "lorahub/flan_t5_large-qasc_qa_with_separated_facts_5",
    "lorahub/flan_t5_large-quoref_Guess_Title_For_Context",
    "lorahub/flan_t5_large-wiki_hop_original_explain_relation",
    "lorahub/flan_t5_large-ropes_prompt_beginning",
    "lorahub/flan_t5_large-gem_e2e_nlg",
    "lorahub/flan_t5_large-race_high_Select_the_best_answer_no_instructions_",
    "lorahub/flan_t5_large-quail_context_question_description_answer_id",
    "lorahub/flan_t5_large-qasc_qa_with_combined_facts_1",
    "lorahub/flan_t5_large-glue_cola",
    "lorahub/flan_t5_large-quail_description_context_question_answer_id",
    "lorahub/flan_t5_large-wiqa_which_of_the_following_is_the_supposed_perturbation",
    "lorahub/flan_t5_large-sciq_Direct_Question_Closed_Book_",
    "lorahub/flan_t5_large-wmt14_translate_fr-en",
    "lorahub/flan_t5_large-quoref_Context_Contains_Answer",
    "lorahub/flan_t5_large-kilt_tasks_hotpotqa_complex_question",
    "lorahub/flan_t5_large-amazon_polarity_negative_or_positive_tone",
    "lorahub/flan_t5_large-amazon_polarity_would_you_buy",
    "lorahub/flan_t5_large-wiki_qa_exercise",
    "lorahub/flan_t5_large-adversarial_qa_dbert_tell_what_it_is",
    "lorahub/flan_t5_large-word_segment",
    "lorahub/flan_t5_large-gem_dart",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_extract_answer",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_title_generation",
    "lorahub/flan_t5_large-ropes_plain_bottom_hint",
    "lorahub/flan_t5_large-wiki_bio_comprehension",
    "lorahub/flan_t5_large-anli_r2",
    "lorahub/flan_t5_large-quail_context_question_answer_description_text",
    "lorahub/flan_t5_large-wiki_hop_original_generate_object",
    "lorahub/flan_t5_large-squad_v1.1",
    "lorahub/flan_t5_large-wiki_qa_Jeopardy_style",
    "lorahub/flan_t5_large-lambada",
    "lorahub/flan_t5_large-quartz_having_read_above_passage",
    "lorahub/flan_t5_large-quartz_use_info_from_question_paragraph",
    "lorahub/flan_t5_large-wiki_bio_key_content",
    "lorahub/flan_t5_large-duorc_SelfRC_answer_question",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_answer_question",
    "lorahub/flan_t5_large-wiki_qa_Topic_Prediction_Question_and_Answer_Pair",
    "lorahub/flan_t5_large-anli_r3",
    "lorahub/flan_t5_large-glue_mnli",
    "lorahub/flan_t5_large-wiki_bio_guess_person",
    "lorahub/flan_t5_large-race_high_Select_the_best_answer_generate_span_",
    "lorahub/flan_t5_large-glue_stsb",
    "lorahub/flan_t5_large-gem_web_nlg_en",
    "lorahub/flan_t5_large-adversarial_qa_droberta_based_on",
    "lorahub/flan_t5_large-duorc_SelfRC_question_answering",
    "lorahub/flan_t5_large-dream_read_the_following_conversation_and_answer_the_question",
    "lorahub/flan_t5_large-duorc_SelfRC_generate_question_by_answer",
    "lorahub/flan_t5_large-definite_pronoun_resolution",
    "lorahub/flan_t5_large-quartz_read_passage_below_choose",
    "lorahub/flan_t5_large-race_middle_Is_this_the_right_answer",
    "lorahub/flan_t5_large-wiqa_effect_with_label_answer",
    "lorahub/flan_t5_large-wiqa_what_might_be_the_first_step_of_the_process",
    "lorahub/flan_t5_large-sciq_Multiple_Choice",
    "lorahub/flan_t5_large-quartz_use_info_from_paragraph_question",
    "lorahub/flan_t5_large-quarel_do_not_use",
    "lorahub/flan_t5_large-quac",
    "lorahub/flan_t5_large-glue_qqp",
    "lorahub/flan_t5_large-quail_no_prompt_text",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_decide_worth_it",
    "lorahub/flan_t5_large-wiqa_effect_with_string_answer",
    "lorahub/flan_t5_large-wiki_hop_original_choose_best_object_interrogative_2",
    "lorahub/flan_t5_large-bool_q",
    "lorahub/flan_t5_large-social_i_qa_Check_if_a_random_answer_is_valid_or_not",
    "lorahub/flan_t5_large-ropes_prompt_bottom_hint_beginning",
    "lorahub/flan_t5_large-newsroom",
    "lorahub/flan_t5_large-ropes_prompt_mix",
    "lorahub/flan_t5_large-quartz_answer_question_based_on",
    "lorahub/flan_t5_large-qasc_qa_with_separated_facts_1",
    "lorahub/flan_t5_large-race_high_Select_the_best_answer",
    "lorahub/flan_t5_large-duorc_ParaphraseRC_movie_director",
    "lorahub/flan_t5_large-amazon_polarity_user_satisfied",
    "lorahub/flan_t5_large-sentiment140",
    "lorahub/flan_t5_large-glue_mrpc",
    "lorahub/flan_t5_large-super_glue_multirc",
    "lorahub/flan_t5_large-quoref_Answer_Test",
    "lorahub/flan_t5_large-wiqa_what_is_the_missing_first_step",
    "lorahub/flan_t5_large-race_middle_Select_the_best_answer_no_instructions_",
    "lorahub/flan_t5_large-snli",
    "lorahub/flan_t5_large-dbpedia_14_pick_one_category_for_the_following_text",
    "lorahub/flan_t5_large-amazon_polarity_Is_this_review_negative",
    "lorahub/flan_t5_large-quarel_testing_students",
    "lorahub/flan_t5_large-glue_qnli",
    "lorahub/flan_t5_large-kilt_tasks_hotpotqa_final_exam",
    "lorahub/flan_t5_large-web_questions_get_the_answer",
    "lorahub/flan_t5_large-duorc_SelfRC_decide_worth_it",
    "lorahub/flan_t5_large-paws_wiki",
    "lorahub/flan_t5_large-social_i_qa_Show_choices_and_generate_index",
    "lorahub/flan_t5_large-duorc_SelfRC_extract_answer",
    "lorahub/flan_t5_large-drop",
    "lorahub/flan_t5_large-adversarial_qa_droberta_answer_the_following_q",
    "lorahub/flan_t5_large-amazon_polarity_Is_this_product_review_positive",
    "lorahub/flan_t5_large-quail_no_prompt_id",
    "lorahub/flan_t5_large-wiki_qa_automatic_system",
    "lorahub/flan_t5_large-sciq_Multiple_Choice_Question_First",
    "lorahub/flan_t5_large-squad_v2.0",
    "lorahub/flan_t5_large-wiqa_does_the_supposed_perturbation_have_an_effect",
    "lorahub/flan_t5_large-wiki_bio_what_content",
    "lorahub/flan_t5_large-duorc_SelfRC_movie_director",
    "lorahub/flan_t5_large-quarel_logic_test",
    "lorahub/flan_t5_large-quartz_answer_question_below",
    "lorahub/flan_t5_large-dbpedia_14_given_list_what_category_does_the_paragraph_belong_to",
    "lorahub/flan_t5_large-amazon_polarity_Is_this_review",
    "lorahub/flan_t5_large-race_middle_Write_a_multi_choice_question_options_given_",
    "lorahub/flan_t5_large-adversarial_qa_dbidaf_tell_what_it_is",
    "lorahub/flan_t5_large-quail_context_description_question_answer_text"
]
