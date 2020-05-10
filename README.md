# GPU-Gram-Scan
GPU Accelerated Gram Scan

implemet merge sort. realized want bitonic

figure out how bitonic works and make sutup of infra forcuda. set up cuda but still confused how will implement on gpu

figure out how to do on gpu with psudocode. get a good idea and now feel like fully understand algo, but do not know yet how to partition sorting network into blocks and threads

figure that out.  got a good idea also wrote some edge cases I need to look out for like non-power-of-2 and how to handle

implement everything but the kernels, getting blocka nd threads down. realized that way more complicated wifuring out getting block size ask mike for help start to devise new way of partitioning blocks and chunks.

ran into errors with cuda and c++... 
ran into errors with cuda and c++... also bug fix in n random and error checking

let's actually do this now. actually somehow did it I think but now need to test

testing the sort. does not work.  when made k=5 realized all the correct comparisons were happening, but not all the correst swaps, think something wrong with comparator

enlist mike and debug