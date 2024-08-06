import pandas as pd

# 'sentences' post-processing
def post_process_sentence(sentence):
    if not isinstance(sentence, str):
        return str(sentence)

    # 1. Remove asterisks (*), double quotes ("), and underscores (_)
    sentence = sentence.replace('*', '').replace('"', '').replace('_', '')

    # 2. Truncate sentence after newline (\n)
    sentence = sentence.split('\n')[0]

    # 3. Remove "The answer is"
    sentence = sentence.replace('The answer is', '').strip()

    # 4. Remove trailing period (.)
    if sentence.endswith('.'):
        sentence = sentence[:-1]

    # 5. Convert "No" to "no"
    sentence = sentence.replace('No', 'no')
    sentence = sentence.replace('No.', 'no')

    # 6. Convert "Yes" to "yes"
    sentence = sentence.replace('Yes', 'yes')
    sentence = sentence.replace('Yes.', 'yes')

    # 7. Convert "None" to "no"
    sentence = sentence.replace('None', 'no')

    return sentence

# Define function to replace empty cells with 'None'
def replace_empty_with_no_answer(cell):
    if pd.isna(cell) or (isinstance(cell, str) and cell.strip() == '') or cell == None or cell == 'None':
        return 'no'
    else:
        return cell
    
# # Function to correct spelling and grammar
# def correct_text(sentence):
#     if not isinstance(sentence, str) or sentence == 'None.':
#         return sentence
# 
#     blob = TextBlob(sentence)
#     corrected_sentence = str(blob.correct())
# 
#     return corrected_sentence

def check_duplicated_id(df):

    duplicate_ids = df[df.duplicated(subset='id', keep=False)]

    ids_sorted_correctly = (df['id'].sort_values().reset_index(drop=True) == df['id']).all()

    if not duplicate_ids.empty:
        print("Duplicate IDs found:")
        print(duplicate_ids)
    else:
        print("No duplicate IDs found.")

    if ids_sorted_correctly:
        print("IDs are in ascending order starting from 0.")
    else:
        print("IDs are not in ascending order starting from 0.")

def check_null_sentences(df):
    null_sentence_ids = df[df['sentences'].isna() | df['sentences'].astype(str).str.strip().isin([''])]['id'].tolist()

    if null_sentence_ids:
        print("IDs with null sentences:")
        print(null_sentence_ids)
    else:
        print("No null sentences found.")

# # Post-process with grammer or word checker
# df['sentences'] = df['sentences'].apply(correct_text)
# print("word corrected")

# # Function to find sentences with multiple sentences
# def check_multiple_sentences(df, col='sentences'):
#     multiple_sentence_ids = []
#     for index, row in df.iterrows():
#         sentence = row[col]
#         if isinstance(sentence, str):
#             # Count sentence terminators (. ? !)
#             sentence_count = sum([sentence.count('\n')])
#             if sentence_count >= 2:
#                 multiple_sentence_ids.append(row['id'])
# 
#     if multiple_sentence_ids:
#         print("IDs with two or more sentences in {} column:".format(col))
#         print(multiple_sentence_ids)
#     else:
#         print("No rows found with two or more sentences in {} column.".format(col))

def check_multiple_sentences(df, col='sentences'):
    multiple_sentence_ids = []
    for index, row in df.iterrows():
        sentence = row[col]
        if isinstance(sentence, str):
            # Count the number of sentences based on newline characters
            sentence_count = sentence.count('\n') + 1  # Adding 1 to count the first sentence
            if sentence_count >= 2:
                multiple_sentence_ids.append(row['id'])
                print(row['id'])

    if multiple_sentence_ids:
        print("IDs with two or more sentences in {} column:".format(col))
        print(multiple_sentence_ids)
    else:
        print("No rows found with two or more sentences in {} column.".format(col))

# Read CSV file
df = pd.read_csv('./output.csv')

# Post-process the 'sentences' column
print("sentence post-processing begin")
df['sentences'] = df['sentences'].apply(post_process_sentence)
print("*, - removed")

# df = df.map(replace_empty_with_no_answer)
df['sentences'] = df['sentences'].apply(replace_empty_with_no_answer)
print("replaced no answer")

# Perform the check
check_multiple_sentences(df, col='sentences')
check_duplicated_id(df)
check_null_sentences(df)
check_multiple_sentences(df, 'sentences')
check_multiple_sentences(df, 'queries')
check_multiple_sentences(df, 'id')

# 6826 col -> 0. not changed

# Save the modified DataFrame to a new CSV file
df.to_csv('./submission.csv', index=False)
print("Modified CSV file has been created successfully.")
