###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import sys

def print_sep(w):
    new_row = ['-' * s for s in w]
    print("|%s|" % '|'.join(new_row))

def print_row(r,w):
    new_row = ['%*s' % (w[i],r[i]) for i in range(len(r))]
    print("|%s|" % '|'.join(new_row))

def make_table(t):
    max_cols=0
    max_width=[]
    for r in t:
        max_cols = max(max_cols,len(r))
        if len(max_width) < max_cols:
            max_width += [0]*(max_cols - len(max_width))
        for i,item in enumerate(r):
            if len(item) > max_width[i]:
                max_width[i] = len(item)
    for r in t:
        if len(r) < max_cols:
            r +=['']*(max_cols - len(r))
    print_row(t[0],max_width)
    print_sep(max_width)
    for r in t[1:]:
        print_row(r,max_width)
    print("")


with open(sys.argv[1],'r') as f:
    table=[]
    for line in f.readlines():
        line = line.strip()
        if line=='' and len(table) > 0:
            make_table(table)
            table=[]
        else:
            table.append([v.replace('_','\\_') for v in line.split('\t')])

if table:
    make_table(table)



