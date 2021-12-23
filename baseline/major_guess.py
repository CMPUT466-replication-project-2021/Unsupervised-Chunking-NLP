from subprocess import run, PIPE

def valid_conll_eval(fname):

	with open(fname, 'r') as file:
		data = file.read()

	pipe = run(["perl", "eval_conll2000_updated.pl"], stdout=PIPE, input=data, encoding='ascii')
	output = pipe.stdout

	tag_acc = float(output.split()[0])
	phrase_f1 = float(output.split()[1])

	print("tag_acc, phrase_f1", tag_acc, phrase_f1)
	return phrase_f1

filename = "temp.txt"
f = open(filename, "w")

test = "../test.txt"
ftest = open(test, "r")
for line in ftest:
    word = line.split()
    if len(word) == 0:
        continue
    f.write("x y " + word[2][0] + " B\n")
f.close()
fscore = valid_conll_eval(filename)