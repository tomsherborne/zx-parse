#!/bin/bash
if [[ $# == 0 ]]; then
	echo "USAGE: execute_sql.sh QUERY_LIST TGT_LOC"
fi
QUERY_LIST=$1
OUTPUT_LOC=$2
> ${OUTPUT_LOC}
while IFS= read -r query
do
    QCLEAN=$(echo "${query}" | sed  "s/ (/(/g")
#    echo ${QCLEAN}
	OUTPUT=$(timeout 1 mysql -u ${USER} atis_new -e "${QCLEAN};" 2> /dev/null)

	if [[ -z $OUTPUT ]]; then
		OUTPUT="###[EMPTY_DENOTATION]###"
	fi
	echo $OUTPUT >> $OUTPUT_LOC
done < $QUERY_LIST
