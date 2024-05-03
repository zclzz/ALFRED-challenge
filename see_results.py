episode_count = 0
success_count = 0
GC_SR = 0
with open('results/shell/log_tests_seen_noappended.txt') as f:
    for line in f:
        # print(line, end='')
        if line.startswith('episode # is'):
            episode_count += 1
        if line.startswith('success is probably True'):
            success_count += 1
        if line.startswith ('goal condition success rate is'):
            GC_SR += float(line.split()[-1])
print('Number of episodes:', episode_count)
print('Estimated number of successful episodes:', success_count)
print('Estimated success rate:', success_count/episode_count)
