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

package qa.qcri.nadeef.service;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.*;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.ProgressReport;
import qa.qcri.nadeef.core.pipeline.CleanExecutor;
import qa.qcri.nadeef.service.thrift.TJobStatus;
import qa.qcri.nadeef.service.thrift.TJobStatusType;
import qa.qcri.nadeef.tools.Logger;

import java.lang.ref.WeakReference;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;

/**
 * NADEEF job scheduler.
 */
public class NadeefJobScheduler {
    private static NadeefJobScheduler instance;
    private static ListeningExecutorService service;
    private static List<String> keys;
    private static ConcurrentMap<String, NadeefJob> runningCleaner;
    private static ConcurrentMap<String, String> runningRules;
    private static String hostname;
    private static Logger tracer = Logger.getLogger(NadeefServiceHandler.class);

    static {
        try {
            hostname = InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            tracer.error("Unknown hostname", e);
            hostname = "127.0.0.1";
        }

        // TODO: the limit is one.
        service = MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(1));
        runningCleaner = Maps.newConcurrentMap();
        runningRules = Maps.newConcurrentMap();
        keys = Collections.synchronizedList(new ArrayList<String>());
    }

    private enum JobType {
        Detect,
        Repair
    }

    /**
     * NadeefJob class represents a runnable job.
     */
    private static class NadeefJob {
        NadeefJob(String key, CleanExecutor executor, JobType type) {
            this.key = key;
            this.executor = executor;
            this.type = type;
        }

        public String key;
        public CleanExecutor executor;
        public JobType type;
    }

    /**
     * Container for executing a clean plan.
     */
    class CleanExecutorCaller implements Callable<String> {
        private WeakReference<NadeefJob> jobRef;

        public CleanExecutorCaller(WeakReference<NadeefJob> jobRef) {
            Preconditions.checkNotNull(jobRef);
            this.jobRef = jobRef;
        }

        public String call() throws Exception {
            NadeefJob job = jobRef.get();
            if (job == null) {
                throw new NullPointerException("Job reference is null in execution.");
            }

            switch (job.type) {
                case Detect:
                    job.executor.detect();
                    break;
                case Repair:
                    job.executor.repair();
                    break;
            }
            return job.key;
        }
    }

    /**
     * Callback function once a clean is done.
     */
    class CleanCallback implements FutureCallback<String> {
        private Logger tracer = Logger.getLogger(CleanCallback.class);

        public void onSuccess(String key) {
            keys.remove(key);
            runningCleaner.remove(key);
            runningRules.remove(key);
        }

        @Override
        public void onFailure(Throwable throwable) {
            tracer.error("FutureCallback failed.");
        }
    }

    /**
     * Singleton access.
     * @return NadeefJobScheduler.
     */
    public synchronized static NadeefJobScheduler getInstance() {
        if (instance == null) {
            instance = new NadeefJobScheduler();
        }
        return instance;
    }

    /**
     * Submits a detection job.
     * @param cleanPlan clean plan.
     * @return job key.
     */
    public String submitDetectJob(CleanPlan cleanPlan) throws Exception {
        NadeefJob job = createNewJob(cleanPlan, JobType.Detect);
        ListenableFuture<String> future =
            service.submit(new CleanExecutorCaller(new WeakReference<>(job)));
        Futures.addCallback(future, new CleanCallback());
        return job.key;
    }

    /**
     * Submits a repair job.
     * @param cleanPlan clean plan.
     * @return job key.
     */
    public String submitRepairJob(CleanPlan cleanPlan) throws Exception {
        NadeefJob job = createNewJob(cleanPlan, JobType.Repair);

        ListenableFuture<String> future =
            service.submit(new CleanExecutorCaller(new WeakReference<>(job)));
        Futures.addCallback(future, new CleanCallback());
        return job.key;
    }

    /**
     * Gets the status of a given job key.
     * @param key job key.
     * @return {@link TJobStatus}.
     */
    // TODO: should we separate Thrift code?
    public TJobStatus getJobStatus(String key) {
        TJobStatus result = new TJobStatus();
        if (!runningCleaner.containsKey(key)) {
            result.setStatus(TJobStatusType.NOTAVAILABLE);
            return result;
        }

        NadeefJob job = runningCleaner.get(key);
        CleanExecutor executor = job.executor;
        double progress = 0f;
        List<ProgressReport> detailProgress = null;
        switch(job.type) {
            case Detect:
                progress = executor.getDetectProgress();
                detailProgress = executor.getDetailDetectProgress();
                break;
            case Repair:
                progress = executor.getRepairProgress();
                detailProgress = executor.getDetailRepairProgress();
                break;
        }

        result.setOverallProgress((int) (progress * 100));
        List<String> names = Lists.newArrayList();
        List<Integer> progresses = Lists.newArrayList();
        for (int i = 0; i < detailProgress.size(); i ++) {
            names.add(detailProgress.get(i).getOperatorName());
            progresses.add((int)detailProgress.get(i).getProgress() * 100);
        }

        result.setNames(names);
        result.setProgress(progresses);
        // a hack to determine whether the job is executing or not.
        if (executor.isRunning()) {
            result.setStatus(TJobStatusType.RUNNING);
        } else {
            result.setStatus(TJobStatusType.WAITING);
        }

        result.setKey(key);
        return result;
    }

    /**
     * Gets the job status of all running jobs.
     * @return the job status of all running jobs.
     */
    public List<TJobStatus> getJobStatus() {
        List<TJobStatus> result = Lists.newArrayList();
        for (String key : keys) {
            result.add(getJobStatus(key));
        }
        return result;
    }

    private static synchronized NadeefJob createNewJob(
        CleanPlan cleanPlan,
        JobType type
    ) throws Exception {
        Preconditions.checkNotNull(cleanPlan);

        String ruleName = cleanPlan.getRule().getRuleName();
        if (runningRules.containsValue(ruleName)) {
            tracer.info("Submitting duplicate rules.");
        }

        String key;

        while (true) {
            key = hostname + "_" + UUID.randomUUID().toString();
            if (!runningCleaner.containsKey(key)) {
                break;
            }
        }
        NadeefJob job =
            new NadeefJob(
                key,
                new CleanExecutor(cleanPlan, cleanPlan.getSourceDBConfig()),
                type
            );

        keys.add(key);
        runningCleaner.put(key, job);
        runningRules.put(key, ruleName);
        return job;
    }
}
