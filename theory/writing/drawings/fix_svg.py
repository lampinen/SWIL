import xml.etree.ElementTree as ET
from copy import deepcopy

filename = "rumelhart_revamped.svg"
output_filename = "rumelhart_revamped_tweaked.svg"

def elements_equal_up_to_id(e1, e2):
    """Stolen/tweaked from https://stackoverflow.com/questions/27549653/python-remove-duplicate-elements-from-xml-tree"""
    if type(e1) != type(e2):
        return False
    if e1.tag != e2.tag: return False
    if e1.text != e2.text: return False
    if e1.tail != e2.tail: return False

    if "id" in e1.attrib:
        e2_attrib_tweak = deepcopy(e2.attrib)
        e2_attrib_tweak["id"] = e1.attrib["id"] 
    if e1.attrib != e2_attrib_tweak: return False
    if len(e1) != len(e2): return False
    return all([elements_equal_up_to_id(c1, c2) for c1, c2 in zip(e1, e2)])

def remove_duplicate_children_recursive(node):
    to_remove = []
    for i, child1 in enumerate(node):
        for j, child2 in enumerate(node):
            if i < j and elements_equal_up_to_id(child1, child2):
                to_remove.append(child2)
    for child in to_remove:
        try:
            node.remove(child)
        except ValueError:
            pass  # already removed.
    for child in node:
        remove_duplicate_children_recursive(child)

tree = ET.parse(filename)
root = tree.getroot()
remove_duplicate_children_recursive(root)

with open(output_filename, "w") as fout:
    tree.write(fout, encoding="utf-8")

