with open('corpus.yml', 'r') as f:
    text = f.read().splitlines()

#line=word1 var[i+1]=word2
with open('current_training_data.txt', 'r') as fp:
    original = fp.read()[:-1]
with open('current_training_data.txt', 'w') as file:
    file.write(original)
    n = text[3:]
    var = enumerate(n)
    for i, line in var:
        if i < len(n)-1 and '- -' in n[i+1]:
            pass
        elif i < len(n)-1 and '-' in n[i+1]:
            if i == 0:
                context = line[4:].replace('"', "'")
                response = n[i+1][4:].replace('"', "'")
                file.write('    ["{0}", "{1}"],'.format(context, response))
            else:
                context = line[4:].replace('"', "'")
                response = n[i+1][4:].replace('"', "'")
                file.write('\n    ["{0}", "{1}"],'.format(context, response))

    file.write('\n]')