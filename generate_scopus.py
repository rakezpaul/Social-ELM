import os
import traceback
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval
import json
from tqdm import tqdm

if __name__ == "__main__":
    # Initialize pybliometrics
    import pybliometrics
    pybliometrics.scopus.init()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Begin script
    for year in range(2018, 2024):
        # Make the folder to store the data for the year
        folder_path = os.path.join(output_dir, str(year))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        try:
            # Get the results
            x = ScopusSearch(
                f'ABS ( "engineered living materials" ) OR TITLE ( "engineered living material" ) AND PUBYEAR = {year}',
                view="STANDARD"
            )
            print(f"Year: {year}, Results count: {len(x.results)}")

            # Check if results exist
            if x.results:
                # Store the results and add the ref_docs key to store each reference
                for doc in tqdm(x.results):
                    try:
                        # Store each result in a file labeled by its Scopus EID
                        doc_dict = doc._asdict()
                        eid = doc_dict["eid"]
                        file_path = os.path.join(folder_path, f"{eid}.json")
                        if not os.path.exists(file_path):
                            # Look up the references / citations for that document
                            document = AbstractRetrieval(eid, view="REF")
                            refs = []
                            # Store the references
                            for ref in document.references:
                                ref_doc = {"doi": ref.doi, "title": ref.title,
                                           "id": ref.id,
                                           "sourcetitle": ref.sourcetitle}
                                refs.append(ref_doc)

                            doc_dict["ref_docs"] = refs
                            # Dump the dictionary to the JSON file
                            with open(file_path, "w") as json_file:
                                json.dump(doc_dict, json_file)
                            print(f"Saved: {file_path}")
                        else:
                            print(f"SKIP (File already exists): {file_path}")
                    except Exception as e:
                        print(f"Error processing document {doc.eid}: {e}")
                        traceback.print_exc()
            else:
                print(f"No results for year {year}.")