import sys

if __name__ == '__main__':
    Usage = "{0} <raw_rating_file> <output_file_dir> <split_count>".format(sys.argv[0])

    if len(sys.argv) != 4:
        print "Invalid argument number. Usage:"
        print Usage
        sys.exit(2)

    raw_rating_file = sys.argv[1]
    output_dir = sys.argv[2]
    split_count = int(sys.argv[3])

    print 'Start split raw rating file:[%s] into [%d] files and save into dir:[%s].' % (raw_rating_file, split_count,
                                                                                       output_dir)
    output_file_writer_list = [None] * split_count
    print len(output_file_writer_list)
    for i in range(0, split_count):
        output_file_writer_list[i] = open("%s/%d" % (output_dir, i), 'w')

    with open(raw_rating_file, 'r') as f:
        line_no = 0
        for line in f:
            if line_no == 0:
                line_no += 1
                continue
            writer_no = line_no % split_count
            current_writer = output_file_writer_list[writer_no]
            current_writer.write(line)
            line_no += 1
            if line_no % 10000 == 0:
                print '%d lines are processed' % line_no
        print 'Total %d lines are split into %d files, every file nearly contains %d lines' % (line_no, split_count,
                                                                                               line_no/split_count)

    for writer in output_file_writer_list:
        writer.close()
