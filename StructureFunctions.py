import pandas as pd
def build_structure(ori, tar, struct_fn, model=None):
    """
    Builds a structure based on the given dataframes and sentence building function
    
    :param ori: original dataframe
    :param tar: target dataframe
    :param struct_fn: sentence building function
    :param model: model to use for tokenization
    :return: two dataframes with the generated sentences
    """
    if struct_fn == build_token_structure:
        # Add special tokens for column names and values
        spec_tokens = {"additional_special_tokens": ["[COL]", "[VAL]"]}
        trans = model._first_module()
        trans.tokenizer.add_special_tokens(spec_tokens)
        trans.auto_model.resize_token_embeddings(len(trans.tokenizer))
        print('Added special tokens')
    elif struct_fn == build_token_structure_with_support or struct_fn == build_token_structure_with_marital_household:
        # Add special tokens for column names, values and support
        spec_tokens = {"additional_special_tokens": ["[COL]", "[VAL]", "[SUP]"]}
        trans = model._first_module()
        trans.tokenizer.add_special_tokens(spec_tokens)
        trans.auto_model.resize_token_embeddings(len(trans.tokenizer))
        print('Added special tokens')

    # Apply the sentence building function to the dataframes
    return ori.apply(lambda r: struct_fn(model=model, r=r, db="o"), axis=1), tar.apply(
        lambda r: struct_fn(model=model, r=r, db="t"), axis=1)

def build_sentence(r, db: str, model=None):
    """
    Returns a sentence based on columns information

    
    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :param model: model to use for tokenization
    :return: a sentence describing the person
    """

    def sex_word(x):
        """
        Returns a description based on sex value

        :param x: sex value ('f' or 'm')
        :return: "female" or "male"        """
        #if pd.isna(x):
        #    return "person"
        return "female" if str(x).strip().lower() == "f" else "male" if str(x).strip().lower() == "m" else "person"

    parts = []

    if pd.isna(r['name_' + db]):
        parts.append('this is an unknown person')
    # name and maiden name
    else:
        if not 'født' in str(r['name_' + db]) and pd.notna(r['name_' + db]):
            parts.append(f"{r['name_' + db]}")

            if not pd.isna(r['maiden_names_' + db]):
                parts.append(f"born {r['maiden_names_' + db]}")
        else:
            parts.append(r['name_' + db])

        # sex
        parts.append(f"is a {sex_word(r['sex_' + db])}")

    # birth year
    if pd.notna(r['birth_year_' + db]):
        parts.append(f"born in {int(r['birth_year_' + db])}")

    # birth place
    if sum(pd.notna([r['birth_town_' + db], r['birth_parish_' + db], r['birth_county_' + db]])) >= 2:
        parts.append("in")
        if pd.notna(r['birth_town_' + db]):
            parts.append(f"{r['birth_town_' + db]}")
        if pd.notna(r['birth_parish_' + db]):
            parts.append(f"{r['birth_parish_' + db]} parish")
        if pd.notna(r['birth_county_' + db]):
            parts.append(f"{r['birth_county_' + db]} county")

    elif r['birth_place_' + db] == 'her i sogn':
        parts.append(f"in {r['event_parish_' + db]}")
    elif pd.notna(r['birth_place_cl_' + db]):
        parts.append(f"in {r['birth_place_cl_' + db]}")
    elif pd.notna(r['birth_place_' + db]):
        parts.append(f"in {r['birth_place_' + db]}")

    # birth country
    if pd.notna(r['birth_country_' + db]):
        parts.append(f"in {r['birth_country_' + db]}")
    sentence = " ".join(parts).strip()

    # Replace Danish words with English
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county"))
    return sentence + "."

def build_sentence_with_support(r, db: str, model=None):
    """
    Returns a sentence based on columns information

    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :param model: model to use
    :return: a sentence describing the person with support information
    """

    def sex_word(x):
        """
        Returns a description based on sex value
        :param x: sex value ('f' or 'm')
        :return: "female", "male" or "person"
        """
        return "female" if str(x).strip().lower() == "f" else "male" if str(x).strip().lower() == "m" else "person"
    parts = []

    if pd.isna(r['name_' + db]):
        parts.append('this is an unknown person')
    # name and maiden name
    else:
        if not 'født' in str(r['name_' + db]) and pd.notna(r['name_' + db]):
            parts.append(f"{r['name_' + db]}")

            if not pd.isna(r['maiden_names_' + db]):
                parts.append(f"born {r['maiden_names_' + db]}")
        else:
            parts.append(r['name_' + db])

        # sex
        parts.append(f"is a {sex_word(r['sex_' + db])}")

    # birth year
    if pd.notna(r['birth_year_' + db]):
        parts.append(f"born in {int(r['birth_year_' + db])}")

    # birth place
    if sum(pd.notna([r['birth_town_' + db], r['birth_parish_' + db], r['birth_county_' + db]])) >= 2:
        parts.append("in")
        if pd.notna(r['birth_town_' + db]):
            parts.append(f"{r['birth_town_' + db]}")
        if pd.notna(r['birth_parish_' + db]):
            parts.append(f"{r['birth_parish_' + db]} parish")
        if pd.notna(r['birth_county_' + db]):
            parts.append(f"{r['birth_county_' + db]} county")

    elif r['birth_place_' + db] == 'her i sogn':
        parts.append(f"in {r['event_parish_' + db]}")
    elif pd.notna(r['birth_place_cl_' + db]):
        s = f"in {r['birth_place_cl_' + db]}"
        s = (s.replace(" a ", " county "))
        parts.append(s)
    elif pd.notna(r['birth_place_' + db]):
        s = f"in {r['birth_place_' + db]}"
        s = (s.replace(" a ", " county "))
        parts.append(s)

    # birth country
    if pd.notna(r['birth_country_' + db]):
        parts.append(f"in {r['birth_country_' + db]}")


    # Support info (male)
    if pd.notna(r['name_fst_non_child_servant_male_' + db]) or pd.notna(r['name_fst_non_child_servant_female_' + db]):
        parts.append(f", and in {int(r['event_year_' + db])}")
        if pd.notna(r['name_fst_non_child_servant_male_' + db]):
            parts.append(f"{r['name_fst_non_child_servant_male_' + db]} is the male head of family,")
            if pd.notna(r['birth_year_fst_non_child_servant_male_' + db]):
                parts.append(f"born in {int(r['birth_year_fst_non_child_servant_male_' + db])}")
            # birth place
            if sum(pd.notna([r['birth_parish_fst_non_child_servant_male_' + db], r['birth_county_fst_non_child_servant_male_' + db]])) == 2:
                parts.append("in")
                if pd.notna(r['birth_parish_fst_non_child_servant_male_' + db]):
                    parts.append(f"{r['birth_parish_fst_non_child_servant_male_' + db]} parish")
                if pd.notna(r['birth_county_fst_non_child_servant_male_' + db]):
                    parts.append(f"{r['birth_county_fst_non_child_servant_male_' + db]} county")

            elif r['birth_place_cl_fst_non_child_servant_male_' + db] == 'her i sogn':
                parts.append(f"in {r['event_parish_' + db]}")
            elif pd.notna(r['birth_place_cl_fst_non_child_servant_male_' + db]):
                s = f"in {r['birth_place_cl_fst_non_child_servant_male_' + db]}"
                s = (s.replace(" a ", " county "))
                parts.append(s)
            parts.append(", and")

        if pd.notna(r['name_fst_non_child_servant_female_' + db]):
            parts.append(f"{r['name_fst_non_child_servant_female_' + db]} is the female head of family,")
            if pd.notna(r['birth_year_fst_non_child_servant_female_' + db]):
                parts.append(f"born in {int(r['birth_year_fst_non_child_servant_female_' + db])}")
            # birth place
            if sum(pd.notna([r['birth_parish_fst_non_child_servant_female_' + db],
                             r['birth_county_fst_non_child_servant_female_' + db]])) == 2:
                parts.append("in")
                if pd.notna(r['birth_parish_fst_non_child_servant_female_' + db]):
                    parts.append(f"{r['birth_parish_fst_non_child_servant_female_' + db]} parish")
                if pd.notna(r['birth_county_fst_non_child_servant_female_' + db]):
                    parts.append(f"{r['birth_county_fst_non_child_servant_female_' + db]} county")

            elif r['birth_place_cl_fst_non_child_servant_female_' + db] == 'her i sogn':
                parts.append(f"in {r['event_parish_' + db]}")
            elif pd.notna(r['birth_place_cl_fst_non_child_servant_female_' + db]):
                s = f"in {r['birth_place_cl_fst_non_child_servant_female_' + db]}"
                s = (s.replace(" a ", " county "))
                parts.append(s)

    # Replace Danish words with English
    sentence = " ".join(parts).strip()
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county")
                .replace(" ,", ",")
                .replace(",,", ",")
                .replace(",.", "."))

    return sentence + "."

def build_sentence_with_marital_household(r, db: str, model=None):
    """
    Returns a sentence based on columns information

    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :param model: model to use for tokenization
    :return: a sentence describing the person with marital and household info
    """

    def sex_word(x):
        """
        Returns a description based on sex value
        :param x: sex value ('f' or 'm')
        :return: "female", "male" or "person"
        """
        return "female" if str(x).strip().lower() == "f" else "male" if str(x).strip().lower() == "m" else "person"

    parts = []

    if pd.isna(r['name_' + db]):
        parts.append('this is an unknown person')
    # name and maiden name
    else:
        if not 'født' in str(r['name_' + db]) and pd.notna(r['name_' + db]):
            parts.append(f"{r['name_' + db]}")

            if not pd.isna(r['maiden_names_' + db]):
                parts.append(f"born {r['maiden_names_' + db]}")
        else:
            parts.append(r['name_' + db])

        # sex
        parts.append(f"is a {sex_word(r['sex_' + db])}")

    # birth year
    if pd.notna(r['birth_year_' + db]):
        parts.append(f"born in {int(r['birth_year_' + db])}")

    # birth place
    if sum(pd.notna([r['birth_town_' + db], r['birth_parish_' + db], r['birth_county_' + db]])) >= 2:
        parts.append("in")
        if pd.notna(r['birth_town_' + db]):
            parts.append(f"{r['birth_town_' + db]}")
        if pd.notna(r['birth_parish_' + db]):
            parts.append(f"{r['birth_parish_' + db]} parish")
        if pd.notna(r['birth_county_' + db]):
            parts.append(f"{r['birth_county_' + db]} county")

    elif r['birth_place_' + db] == 'her i sogn':
        parts.append(f"in {r['event_parish_' + db]}")
    elif pd.notna(r['birth_place_cl_' + db]):
        parts.append(f"in {r['birth_place_cl_' + db]}")
    elif pd.notna(r['birth_place_' + db]):
        parts.append(f"in {r['birth_place_' + db]}")

    # birth country
    if pd.notna(r['birth_country_' + db]):
        parts.append(f"in {r['birth_country_' + db]}")

    # marital and household info
    if pd.notna(r['marital_status_' + db]) or pd.notna(r['household_position_' + db]):
        parts.append(f", and is by the year")
        if pd.notna(r['event_year_' + db]):
            parts.append(f"{int(r['event_year_' + db])}")

        if pd.notna(r['marital_status_' + db]):
            status = ['ugift', 'gift', 'enke', 'skilt']
            if r['marital_status_' + db] in status:
                idx = status.index(r['marital_status_' + db])
                status_en = ['single', 'married', 'widowed', 'divorced']
                parts.append(f"{status_en[idx]}")
                if pd.notna(r['household_position_' + db]):
                    parts.append("and")
        if pd.notna(r['household_position_' + db]):
            position = ['hendes barn', 'kone', 'tjeneste', 'barn', 'andet', 'husfader', 'husmoder', 'hans barn']
            if r['household_position_' + db] in position:
                idx = position.index(r['household_position_' + db])
                position_en = ['a child', 'the wife', 'a servant', 'a child', 'a part of', 'the father', 'the mother', 'a child']
                parts.append(f"{position_en[idx]} of the household")


    sentence = " ".join(parts).strip()
    # Replace Danish words with English
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county")
                .replace(" ,", ","))
    return sentence + "."

def build_token_structure(model, r, db: str):
    """
    Returns a token structured sentence based on columns information

    :param model: model to use
    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :return: a token structured sentence of person
    """

    def sex_word(x):
        """
        Returns a description based on sex value
        :param x: sex value ('f' or 'm')
        :return: "female", "male" or "person"
        """
        return "female" if str(x).strip().lower() == "f" else "male" if str(x).strip().lower() == "m" else "person"

    ###################
    # Build structure #
    ###################
    parts = ['[COL] name [VAL]']

    if pd.isna(r['name_' + db]):
        parts.append('unknown')
    # name and maiden name
    else:
        parts.append(f"{r['name_' + db]}")

        if not pd.isna(r['maiden_names_' + db]):
            parts.append(f"born {r['maiden_names_' + db]}")

    # sex
    parts.append(f"[COL] sex [VAL] {sex_word(r['sex_' + db])}")

    # birth year
    parts.append("[COL] birth year [VAL]")
    if pd.notna(r['birth_year_' + db]):
        parts.append(f"{int(r['birth_year_' + db])}")

    # birth place
    parts.append("[COL] birth place [VAL]")
    if sum(pd.notna([r['birth_town_' + db], r['birth_parish_' + db], r['birth_county_' + db]])) >= 2:
        if pd.notna(r['birth_town_' + db]):
            parts.append(f"{r['birth_town_' + db]}")
        if pd.notna(r['birth_parish_' + db]):
            parts.append(f"{r['birth_parish_' + db]}")
        if pd.notna(r['birth_county_' + db]):
            parts.append(f"{r['birth_county_' + db]}")

    elif r['birth_place_' + db] == 'her i sogn':
        parts.append(f"{r['event_parish_' + db]}")
    elif pd.notna(r['birth_place_cl_' + db]):
        parts.append(f"{r['birth_place_cl_' + db]}")
    elif pd.notna(r['birth_place_' + db]):
        parts.append(f"{r['birth_place_' + db]}")

    # birth country
    parts.append("[COL] birth country [VAL]")
    if pd.notna(r['birth_country_' + db]):
        parts.append(f"{r['birth_country_' + db]}")
    else:
        parts.append("denmark")

    sentence = " ".join(parts).strip()
    # Replace Danish words with English
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county"))

    return sentence

def build_token_structure_with_support(model, r, db: str):
    """
    Returns a token structured sentence based on columns information

    :param model: model to use
    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :return: a token structured sentence of person with support info
    """

    ###################
    # Build structure #
    ###################
    parts = [build_token_structure(model, r, db)]
    
    # Support info (male)
    parts.append(f"[SUP] first male [COL] first male name [VAL] {r['name_fst_non_child_servant_male_' + db]}")
    parts.append(f"[COL] first male birth year [VAL]")
    try:
        parts.append(f"{int(r['birth_year_fst_non_child_servant_male_' + db])}")
    except ValueError:
        parts.append("unknown")
    # birth place
    parts.append(f"[COL] first male birth place [VAL]")
    if sum(pd.notna([r['birth_parish_fst_non_child_servant_male_' + db],
                     r['birth_county_fst_non_child_servant_male_' + db]])) == 2:
        if pd.notna(r['birth_parish_fst_non_child_servant_male_' + db]):
            parts.append(f"{r['birth_parish_fst_non_child_servant_male_' + db]} parish")
        if pd.notna(r['birth_county_fst_non_child_servant_male_' + db]):
            parts.append(f"{r['birth_county_fst_non_child_servant_male_' + db]} county")

    elif r['birth_place_cl_fst_non_child_servant_male_' + db] == 'her i sogn':
        parts.append(f"{r['event_parish_' + db]}")
    else:
        s = f"{r['birth_place_cl_fst_non_child_servant_male_' + db]}"
        s = (s.replace(" a ", " county "))
        parts.append(s)

    # Support info (female)
    parts.append(f"[SUP] first female [COL] first female name [VAL] {r['name_fst_non_child_servant_female_' + db]}")
    parts.append(f"[COL] first female birth year [VAL]")
    try:
        parts.append(f"{int(r['birth_year_fst_non_child_servant_female_' + db])}")
    except ValueError:
        parts.append("unknown")
    # birth place
    parts.append(f"[COL] first female birth place [VAL]")
    if sum(pd.notna([r['birth_parish_fst_non_child_servant_female_' + db],
                     r['birth_county_fst_non_child_servant_female_' + db]])) == 2:
        if pd.notna(r['birth_parish_fst_non_child_servant_female_' + db]):
            parts.append(f"{r['birth_parish_fst_non_child_servant_female_' + db]} parish")
        if pd.notna(r['birth_county_fst_non_child_servant_female_' + db]):
            parts.append(f"{r['birth_county_fst_non_child_servant_female_' + db]} county")

    elif r['birth_place_cl_fst_non_child_servant_female_' + db] == 'her i sogn':
        parts.append(f"{r['event_parish_' + db]}")
    else:
        s = f"{r['birth_place_cl_fst_non_child_servant_female_' + db]}"
        s = (s.replace(" a ", " county "))
        parts.append(s)

    sentence = " ".join(parts).strip()
    # Replace Danish words with English
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county")
                .replace(" nan", " unknown "))

    return sentence

def build_token_structure_with_marital_household(model, r, db: str):
    """
    Returns a token structured sentence based on columns information

    :param model: model to use
    :param r: row of a dataframe
    :param db: database suffix ('o' or 't')
    :return: a token structured sentence of person with marital and household info
    """
    ###################
    # Build structure #
    ###################
    parts = [build_token_structure(model, r, db)]

    # Household and marital info
    parts.append(f"[SUP] status [COL] event year [VAL]")
    if pd.notna(r['event_year_' + db]):
        parts.append(f"{int(r['event_year_' + db])}")
    else:
        parts.append("unknown")

    parts.append(f"[COL] marital status [VAL]")
    status = ['ugift', 'gift', 'enke', 'skilt']
    if r['marital_status_' + db] in status:
        idx = status.index(r['marital_status_' + db])
        status_en = ['single', 'married', 'widowed', 'divorced']
        parts.append(f"{status_en[idx]}")
    else:
        parts.append("unknown")

    parts.append(f"[COL] household position [VAL]")
    position = ['hendes barn', 'kone', 'tjeneste', 'barn', 'andet', 'husfader', 'husmoder', 'hans barn']
    if r['household_position_' + db] in position:
        idx = position.index(r['household_position_' + db])
        position_en = ['child', 'wife', 'servant', 'child', 'unknown', 'father', 'mother',
                       'child']
        parts.append(f"{position_en[idx]}")
    else:
        parts.append("unknown")

    sentence = " ".join(parts).strip()
    # Replace Danish words with English
    sentence = (sentence
                .replace("født", "born")
                .replace(" sogn", " parish")
                .replace(" amt", " county")
                .replace(" nan", " unknown"))

    return sentence