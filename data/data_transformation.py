import json
import csv

def get_catalogs():
    catalogs = {}
    terms = ["s26", "f25", "s25", "i25", "f24", "s24", "f23", "s23", "f22"]
    for term in terms:
        with open(f"data/{term}.json", "r") as f:
            term_data = json.load(f)
            catalogs[term] = term_data["classes"]
    return catalogs

def combine_catalogs(catalogs):
    subjects = {}
    for term, catalog in catalogs.items():
        for subject, listing in catalog.items():
            if subject in subjects:
                subjects[subject]["offered"].append(term)
            else:
                subjects[subject] = listing
                subjects[subject]["offered"] = [term]
                subjects[subject]["taken_by"] = []
    return subjects

def combine_wtw(subjects, major):
    with open(f"data/{major}_subjects.csv", "r") as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            subject, total, y1, y2, y3, y4 = line
            if subject not in subjects:
                continue
            if y1:
                subjects[subject]["taken_by"].append(f"{major} freshmen")
            if y2:
                subjects[subject]["taken_by"].append(f"{major} sophomores")
            if y3:
                subjects[subject]["taken_by"].append(f"{major} juniors")
            if y4:
                subjects[subject]["taken_by"].append(f"{major} seniors")
    
if __name__ == "__main__":
    catalogs = get_catalogs()
    subjects = combine_catalogs(catalogs)
    for major in ["6-1", "6-2", "6-3", "6-4", "6-7", "6-14"]:
        combine_wtw(subjects, major)
    with open("data/combined_catalog.json", "w") as f:
        json.dump(subjects, f)
            