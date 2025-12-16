#!/bin/bash

if ! command -v inotifywait &>/dev/null
then
	echo "inotify-tools package is required!"
	exit 1
fi

#Requires inotify-tools
while true
do
	inotifywait -e modify $(find . -name "*.tex" -or -name "*.sty")
	make
done
