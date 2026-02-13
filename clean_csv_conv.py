import csv
import re

# Input and output file paths
input_file = "sequence.txt"
output_file = "sequenceDNA1.csv"

# List to store parsed records
records = []

# Read and split FASTA-like entries
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read().strip().split("\n>")

for entry in content:
    entry = entry.strip()
    if not entry:
        continue

    # Split into header and sequence
    lines = entry.split("\n")
    header = lines[0]
    sequence = "".join(lines[1:]).replace("\n", "").strip()

    # Extract info using regex
    locus_tag = re.search(r"\[locus_tag=(.*?)\]", header)
    protein = re.search(r"\[protein=(.*?)\]", header)
    protein_id = re.search(r"\[protein_id=(.*?)\]", header)
    location = re.search(r"\[location=(.*?)\]", header)
    gbkey = re.search(r"\[gbkey=(.*?)\]", header)

    records.append({
        "locus_tag": locus_tag.group(1) if locus_tag else "",
        "protein": protein.group(1) if protein else "",
        "protein_id": protein_id.group(1) if protein_id else "",
        "location": location.group(1) if location else "",
        "gbkey": gbkey.group(1) if gbkey else "",
        "sequence": sequence
    })

# Write cleaned data to CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["locus_tag", "protein", "protein_id", "location", "gbkey", "sequence"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"✅ Cleaned and converted '{input_file}' → '{output_file}' successfully!")
