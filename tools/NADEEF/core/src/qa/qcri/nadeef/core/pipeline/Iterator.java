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
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.*;
import qa.qcri.nadeef.core.datamodel.IteratorBlockingQueue;
import qa.qcri.nadeef.core.datamodel.PairTupleRule;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Table;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.HashSet;
import java.util.concurrent.*;

/**
 * Iterator.
 */
public class Iterator extends Operator<Collection<Table>, Boolean> {
    //<editor-fold desc="Private members">
    private static final int MAX_THREAD_NUM = Runtime.getRuntime().availableProcessors();
    private int blockCount;

    //</editor-fold>

    public Iterator(ExecutionContext context) {
        super(context);
    }

    //<editor-fold desc="IteratorCallable and Callback Class">

    /**
     * Callback class for progress report.
     */
    class IteratorCallback implements FutureCallback<Integer> {
        private int tableSize;

        public IteratorCallback(int tableSize) {
            this.tableSize = tableSize;
        }
        @Override
        public void onSuccess(Integer integer) {
            synchronized (Iterator.class) {
                blockCount ++;
                setPercentage(blockCount / tableSize);
            }
        }

        @Override
        public void onFailure(Throwable throwable) {}
    }

    /**
     * IteratorCallable is a {@link Callable} class for iteration operation on each block.
     */
    class IteratorCallable<T> implements Callable<Integer> {
        private IteratorBlockingQueue iteratorBlockingQueue;
        private WeakReference<T> ref;
        private WeakReference<ConcurrentMap<String, HashSet<Integer>>> newTupleRef;
        private Rule rule;

        IteratorCallable(
            T tables,
            Rule rule,
            ConcurrentMap<String, HashSet<Integer>> newTuples
        ) {
            this.newTupleRef = new WeakReference<>(newTuples);
            this.ref = new WeakReference<>(tables);
            this.iteratorBlockingQueue = new IteratorBlockingQueue();
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
            T instance = ref.get();
            if (instance == null) {
                throw new RuntimeException("Tables have been freed.");
            }

            Collection<Table> value;
            if (Table.class.isAssignableFrom(instance.getClass())) {
                value = Lists.newArrayList((Table)instance);
            } else {
                value = (Collection<Table>)instance;
            }

            ConcurrentMap<String, HashSet<Integer>> newTuples = newTupleRef.get();
            if (newTuples == null || newTuples.size() == 0 || rule.hasOwnIterator()) {
                rule.iterator(value, iteratorBlockingQueue);
            } else {
                rule.iterator(value, newTuples, iteratorBlockingQueue);
            }
            iteratorBlockingQueue.flush();

            // return the tuple total count
            int size = 0;
            for (Table table : value) {
                size += table.size();
            }
            return size;
        }
    }
    //</editor-fold>

    /**
     * Iterator operator execution.
     *
     * @param blocks input tables.
     * @return iteration output.
     */
    @Override
    @SuppressWarnings("unchecked")
    public Boolean execute(Collection<Table> blocks) {
        Logger tracer = Logger.getLogger(Iterator.class);
        ThreadFactory factory =
            new ThreadFactoryBuilder().setNameFormat("iterator-pool-%d").build();
        ExecutorService executor = Executors.newFixedThreadPool(MAX_THREAD_NUM, factory);
        ListeningExecutorService service = MoreExecutors.listeningDecorator(executor);

        Stopwatch stopwatch = Stopwatch.createStarted();
        blockCount = 0;

        ExecutionContext context = getCurrentContext();
        Rule rule = context.getRule();
        try {
            if (rule.supportTwoTables()) {
                // Rule runs on two tables.
                ListenableFuture<Integer> future =
                    service.submit(new IteratorCallable(blocks, rule, context.getNewTuples()));
                blockCount ++;
                setPercentage(1.0f);
            } else {
                // Rule runs on each table.
                for (Table table : blocks) {
                    ListenableFuture<Integer> future =
                        service.submit(new IteratorCallable(table, rule, context.getNewTuples()));
                    Futures.addCallback(future, new IteratorCallback(blocks.size()));
                }
            }

            // wait until all the tasks are finished
            service.shutdown();
            while (!service.awaitTermination(10l, TimeUnit.MINUTES));

            // recycle the collection when dealing with pairs. This is mainly used to remove refs.
            if (rule instanceof PairTupleRule) {
                for (Table block : blocks) {
                    block.recycle();
                }
            }

            // mark the end of the iteration output
            IteratorBlockingQueue.markEnd();
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
        return true;
    }

    /**
     * Interrupt is called when the Iterator is stopped in the middle. A typical scenario is
     * that there is exception happened during iterator execution.
     */
    @Override
    public void interrupt() {
        IteratorBlockingQueue.markEnd();
    }

    /**
     * Reset is called before iterator starts to execute.
     */
    @Override
    public void reset() {
        IteratorBlockingQueue.clear();
    }
}
