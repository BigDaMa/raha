/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.test.core;

import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import org.junit.Assert;

import java.util.concurrent.*;

public final class IteratorBenchmark {
    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Invalid inputs.");
            System.exit(1);
        }

        int numThread = Integer.parseInt(args[0]);
        int tryCount = Integer.parseInt(args[1]);
        long[] elapsedTime = new long[tryCount];
        int[] sizes = new int[tryCount];

        for (int i = 0; i < tryCount; i ++) {
            Stopwatch stopwatch = Stopwatch.createStarted();
            ThreadFactory factory =
                new ThreadFactoryBuilder().setNameFormat("thread#" + numThread + "-%d").build();
            ExecutorService executor = Executors.newFixedThreadPool(numThread, factory);
            ConcurrentLinkedQueue<Long> queue = new ConcurrentLinkedQueue<>();

            for (int t = 0; t < 5; t ++) {
                executor.submit(() -> {
                    for (int j = 0; j < 2000; j ++)
                        for (int k = j + 1; k < 2000; k ++)
                            queue.offer((long)j << 32 | k);
                });
            }
            executor.shutdown();
            try {
                executor.awaitTermination(10, TimeUnit.MINUTES);
            } catch (Exception ex) {
                Assert.fail(ex.getMessage());
            }

            elapsedTime[i] = stopwatch.elapsed(TimeUnit.MILLISECONDS);
            sizes[i] = queue.size();
        }

        for (int i = 0; i < tryCount; i ++) {
            System.out.println(sizes[i] + "," + elapsedTime[i]);
        }
    }
}
