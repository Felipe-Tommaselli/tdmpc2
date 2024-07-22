#!/bin/bash

python3 train.py 2>&1 | grep -v -E "XML Attribute\[version\] in element\[sdf\] not defined in SDF, ignoring\.|TIFFFetchNormalTag: Warning, ASCII value for tag \"Artist\" does not end in null byte\.|TIFFFetchNormalTag: Warning, Incompatible type for \"RichTIFFIPTC\"; tag ignored\.|TIFFFieldWithTag: Internal error, unknown tag"
