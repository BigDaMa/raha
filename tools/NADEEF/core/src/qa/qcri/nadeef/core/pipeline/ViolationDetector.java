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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.*;
import qa.qcri.nadeef.core.datamodel.*;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

/**
 * Wrapper class for executing the violation detection.
 */
public class ViolationDetector
    extends Operator<Optional, Collection<Violation>> {
    private static final int MAX_THREAD_NUM = Runtime.getRuntime().availableProcessors();

    private Collection<Violation> resultCollection;
    private ListeningExecutorService service;

    private int totalThreadCount;
    private int finishedThreadCount;
    private int detectCount;

    /**
     * Violation detector constructor.
     */
    public ViolationDetector(ExecutionContext context) {
        super(context);
        resultCollection = Lists.newArrayList();
        ThreadFactory factory = new ThreadFactoryBuilder().setNameFormat("detect-pool-%d").build();
        service =
            MoreExecutors.listeningDecorator(
                Executors.newFixedThreadPool(MAX_THREAD_NUM, factory)
            );
    }

    /**
     * Callback class for progress report.
     */
    class DetectorCallback implements FutureCallback<Integer> {
        @Override
        public void onSuccess(Integer integer) {
            synchronized (ViolationDetector.class) {
                finishedThreadCount ++;
                detectCount += integer;
                setPercentage(finishedThreadCount / totalThreadCount);
            }
        }

        @Override
        public void onFailure(Throwable throwable) {}
    }

    /**
     * Detector callable class.
     */
    class Detector implements Callable<Integer> {
        private List<Object> tupleList;
        private Rule rule;

        public Detector(List<Object> tupleList, Rule rule) {
            this.tupleList = Preconditions.checkNotNull(tupleList);
            this.rule = Preconditions.checkNotNull(rule);
        }

        /**
         * Computes a result, or throws an exception if unable to do so.
         *
         * @return detection count
         * @throws Exception if unable to compute a result
         */
        @Override
        @SuppressWarnings("unchecked")
        public Integer call() throws Exception {
            int count = 0;
            Collection<Violation> result = Lists.newArrayList();
            for (int i = 0; i < tupleList.size(); i ++) {
                Object item = tupleList.get(i);

                // breakage in the buffer
                if (item == null) {
                    continue;
                }

                count ++;
                Collection<Violation> violations = null;
                try {
                    if (rule instanceof SingleTupleRule) {
                        Tuple tuple = (Tuple)item;
                        violations = rule.detect(tuple);
                    } else if (rule instanceof PairTupleRule) {
                        TuplePair pair = (TuplePair)item;
                        violations = rule.detect(pair);
                    } else {
                        Table collection = (Table)item;
                        violations = rule.detect(collection);
                    }

                    if (violations != null && violations.size() > 0) {
                        result.addAll(violations);
                    }
                } catch (Exception ex) {
                    Logger tracer = Logger.getLogger(this.getClass());
                    tracer.error("Exception inside detect method.", ex);
                }
            }

            synchronized (ViolationDetector.class) {
                resultCollection.addAll(result);
            }

            // This is to reclaim the memory back.
            tupleList = null;
            return count;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    @SuppressWarnings("unchecked")
    public Collection<Violation> execute(Optional emptyInput) throws Exception {
        detectCount = 0;
        totalThreadCount = 0;
        finishedThreadCount = 0;

        Rule rule = getCurrentContext().getRule();
        IteratorBlockingQueue iteratorBlockingQueue = new IteratorBlockingQueue();
        resultCollection.clear();
        List<Object> tupleList;
        Stopwatch stopwatch = Stopwatch.createStarted();
        List<ListenableFuture<Integer>> futures = Lists.newArrayList();
        while (true) {
            tupleList = iteratorBlockingQueue.poll();
            if (tupleList.size() == 0) {
                break;
            }

            totalThreadCount ++;
            ListenableFuture<Integer> future = service.submit(new Detector(tupleList, rule));
            futures.add(future);
            Futures.addCallback(future, new DetectorCallback());
        }

        // wait until all the futures finished
        for (ListenableFuture<Integer> future : futures) {
            future.get();
        }

        PerfReport.appendMetric(
            PerfReport.Metric.DetectTimeOnly,
            stopwatch.elapsed(TimeUnit.MILLISECONDS)
        );
        PerfReport.appendMetric(PerfReport.Metric.DetectCount, detectCount);
        PerfReport.appendMetric(PerfReport.Metric.DetectThreadCount, totalThreadCount);
        stopwatch.stop();
        return resultCollection;
    }

    @Override
    public void finalize() throws Throwable {
        super.finalize();
        if (!service.isShutdown()) {
            service.shutdownNow();
        }
    }
}