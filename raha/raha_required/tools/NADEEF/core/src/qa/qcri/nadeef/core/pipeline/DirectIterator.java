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

package qa.qcri.nadeef.core.pipeline;

import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import qa.qcri.nadeef.core.datamodel.NonBlockingCollectionIterator;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Table;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.concurrent.*;

public class DirectIterator extends Operator<Collection<Table>, java.util.Iterator<Violation>> {
    private static final int MAX_THREAD_NUM = Runtime.getRuntime().availableProcessors();
    public DirectIterator(ExecutionContext context) {
        super(context);
    }

    /**
     * IteratorCallable is a {@link Callable} class for iteration operation on each block.
     */
    class IteratorCallable implements Callable<Integer> {
        private DirectIteratorResultHandler directIteratorResultHandler;
        private Collection<Table> tables;
        private ConcurrentMap<String, HashSet<Integer>> newTuples;
        private Rule rule;

        IteratorCallable(
            Collection<Table> tables,
            Rule rule,
            ConcurrentMap<String, HashSet<Integer>> newTuples,
            NonBlockingCollectionIterator<Violation> outputIterator
        ) {
            this.newTuples = newTuples;
            this.tables = tables;
            this.directIteratorResultHandler =
                new DirectIteratorResultHandler(rule, outputIterator);
            this.rule = rule;
        }

        /**
         * Computes a result, or throws an exception if unable to do so.
         *
         * @return computed result
         * @throws Exception if unable to compute a result
         */
        @Override
        @SuppressWarnings("unchecked")
        public Integer call() throws Exception {
            if (newTuples == null || newTuples.size() == 0 || rule.hasOwnIterator()) {
                rule.iterator(tables, directIteratorResultHandler);
            } else {
                rule.iterator(tables, newTuples, directIteratorResultHandler);
            }
            return 0;
        }
    }

    @Override
    protected java.util.Iterator<Violation> execute(Collection<Table> blocks) throws Exception {
        Logger tracer = Logger.getLogger(DirectIterator.class);
        ThreadFactory factory =
            new ThreadFactoryBuilder().setNameFormat("iterator-#" + MAX_THREAD_NUM + "-%d").build();
        ExecutorService executor = Executors.newFixedThreadPool(MAX_THREAD_NUM, factory);
        Stopwatch stopwatch = Stopwatch.createStarted();

        ExecutionContext context = getCurrentContext();
        Rule rule = context.getRule();
        NonBlockingCollectionIterator<Violation> output = new NonBlockingCollectionIterator<>();
        try {
            if (rule.supportTwoTables()) {
                // Rule runs on two tables.
                executor.submit(
                    new IteratorCallable(blocks, rule, context.getNewTuples(), output));
            } else {
                // Rule runs on each table.
                for (Table table : blocks)
                    executor.submit(
                        new IteratorCallable(
                            Arrays.asList(table),
                            rule,
                            context.getNewTuples(),
                            output
                        )
                    );
            }

            // wait until all the tasks are finished
            executor.shutdown();
            while (!executor.awaitTermination(10l, TimeUnit.MINUTES));
        } catch (InterruptedException ex) {
            tracer.error("Iterator is interrupted.", ex);
        } finally {
            executor.shutdown();
        }

        PerfReport.appendMetric(
            PerfReport.Metric.IteratorTime,
            stopwatch.elapsed(TimeUnit.MILLISECONDS)
        );

        stopwatch.stop();
        return output;
    }
}
