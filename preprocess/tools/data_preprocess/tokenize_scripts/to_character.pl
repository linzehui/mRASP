#!/usr/bin/env perl

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

use utf8;
use warnings;
use FindBin qw($RealBin);
use strict;
use Time::HiRes;

if  (eval {require Thread;1;}) {
  #module loaded
  Thread->import();
}

my $mydir = "$RealBin/";

my $language = "zh";
my $HELP = 0;
my $NUM_THREADS = 1;
my $NUM_SENTENCES_PER_THREAD = 100000;


while (@ARGV)
{
	$_ = shift;
	/^-b$/ && ($| = 1, next);
	/^-h$/ && ($HELP = 1, next);
	/^-l$/ && ($language = shift, next);
  # Option to add list of regexps to be protected
  	/^-threads$/ && ($NUM_THREADS = int(shift), next);
	/^-lines$/ && ($NUM_SENTENCES_PER_THREAD = int(shift), next);
}

# print help message
if ($HELP)
{
	print "Usage ./to_character.pl -l zh (-threads 4) < textfile > charfile\n";
        print "Options:\n";
        print "  -l     ... the language.\n";
	exit;
}

if ($language ne "zh" && $language ne "ja" && $language ne "ko"){
    print "Not supported language: ".$language."\n";
    exit;
}

my @batch_sentences = ();
my @thread_list = ();
my $count_sentences = 0;

if ($NUM_THREADS > 1)
{# multi-threading tokenization
    while(<STDIN>)
    {
        $count_sentences = $count_sentences + 1;
        push(@batch_sentences, $_);
        if (scalar(@batch_sentences)>=($NUM_SENTENCES_PER_THREAD*$NUM_THREADS))
        {
            # assign each thread work
            for (my $i=0; $i<$NUM_THREADS; $i++)
            {
                my $start_index = $i*$NUM_SENTENCES_PER_THREAD;
                my $end_index = $start_index+$NUM_SENTENCES_PER_THREAD-1;
                my @subbatch_sentences = @batch_sentences[$start_index..$end_index];
                my $new_thread = new Thread \&tokenize_batch, @subbatch_sentences;
                push(@thread_list, $new_thread);
            }
            foreach (@thread_list)
            {
                my $tokenized_list = $_->join;
                foreach (@$tokenized_list)
                {
                    print $_;
                }
            }
            # reset for the new run
            @thread_list = ();
            @batch_sentences = ();
        }
    }
    # the last batch
    if (scalar(@batch_sentences)>0)
    {
        # assign each thread work
        for (my $i=0; $i<$NUM_THREADS; $i++)
        {
            my $start_index = $i*$NUM_SENTENCES_PER_THREAD;
            if ($start_index >= scalar(@batch_sentences))
            {
                last;
            }
            my $end_index = $start_index+$NUM_SENTENCES_PER_THREAD-1;
            if ($end_index >= scalar(@batch_sentences))
            {
                $end_index = scalar(@batch_sentences)-1;
            }
            my @subbatch_sentences = @batch_sentences[$start_index..$end_index];
            my $new_thread = new Thread \&tokenize_batch, @subbatch_sentences;
            push(@thread_list, $new_thread);
        }
        foreach (@thread_list)
        {
            my $tokenized_list = $_->join;
            foreach (@$tokenized_list)
            {
                print $_;
            }
        }
    }
}
else
{# single thread only
    while(<STDIN>)
    {
        print &tokenize($_);
    }
}

#####################################################################################
# subroutines afterward

# tokenize a batch of texts saved in an array
# input: an array containing a batch of texts
# return: another array containing a batch of tokenized texts for the input array
sub tokenize_batch
{
    my(@text_list) = @_;
    my(@tokenized_list) = ();
    foreach (@text_list)
    {
        push(@tokenized_list, &tokenize($_));
    }
    return \@tokenized_list;
}

# the actual tokenize function which tokenizes one input string
# input: one string
# return: the tokenized string for the input string
sub tokenize
{
    my($text) = @_;

    chomp($text);

    my $final = "";
    my @wordlist=split(//,$text);
    for (my $i=0; $i<=$#wordlist; $i++){
        if(&is_character($wordlist[$i]) == 1){
            $final.=" ".$wordlist[$i]." ";
        } else{
            $final.=$wordlist[$i];
        }
    }
#    $final =~ s/[\{-\~\[-\` -\&\(-\+\:-\@\/]/ $1 /g;
    $final =~ s/([^0-9])([\.,])/$1 $2 /g;
    $final =~ s/([\.,])([^0-9])/ $1 $2/g;
    $final =~ s/([0-9])(-)/$1 $2 /g;
    $final =~ s/\s+/ /g;
    $final =~ s/^\s+//g;
    $final =~ s/\s+$//g;
    return $final."\n";

}


sub is_character
{
    my($text) = @_;

    chomp($text);

    if($text =~ /[\x{2E80}-\x{9FFF}]/u || $text =~ /[\x{A000}-\x{A4FF}]/u || $text =~ /[\x{AC00}-\x{D7FF}]/u || $text =~ /[\x{F900}-\x{FAFF}]/u){
        return 1;
    } else{
        return 0;
    }
}

