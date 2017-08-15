import os
import pymarc
file_name = "all_dewey_dump.xml"
full_file = os.path.abspath(os.path.join('dewey_dump',file_name))


#nyttig error:  https://github.com/edsu/pymarc/issues/73
klassebetegnelser = []
deweynr = []
with open(full_file, 'rb') as fh:
    for record in pymarc.parse_xml_to_array(fh):
        if "153" in record:
            if "j" in record["153"] and "a" in record["153"]:
                print(record['153']['j'])
                print(record['153']['a'])
                klassebetegnelser.append(record['153']['j'])
                deweynr.append(record['153']['a'])
print(len(klassebetegnelser))
print(len(deweynr))

zipped = zip(deweynr,klassebetegnelser)
print(list(zipped))
#print(records)
