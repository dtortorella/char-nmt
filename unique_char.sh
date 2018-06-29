#!/bin/bash

perl -C -ne'print grep {!$a{$_}++} /\X/g' "$@"