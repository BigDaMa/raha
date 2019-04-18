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

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.ProgressReport;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.DBInstaller;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.HashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * CleanPlan execution logic. It assembles the right pipeline based on the clean plan and
 * drives the cleaning execution.
 */
public class CleanExecutor {

    //<editor-fold desc="Private fields">
    private static Logger tracer = Logger.getLogger(CleanExecutor.class);
    private CleanPlan cleanPlan;
    private NodeCacheManager cacheManager;
    private Flow queryFlow;
    private Flow detectFlow;
    private Flow repairFlow;
    private DBConnectionPool connectionPool;
    private ExecutionContext context;
    //</editor-fold>

    //<editor-fold desc="Constructor / Deconstructor">

    /**
     * Constructor. Use default NADEEF default config as DB config.
     * @param cleanPlan input {@link CleanPlan}.
     */
    public CleanExecutor(CleanPlan cleanPlan) throws Exception {
        this(cleanPlan, NadeefConfiguration.getDbConfig());
    }

    /**
     * Constructor.
     * @param cleanPlan input {@link CleanPlan}.
     * @param dbConfig meta data dbconfig.
     */
    public CleanExecutor(CleanPlan cleanPlan, DBConfig dbConfig) throws Exception {
        this.cleanPlan = Preconditions.checkNotNull(cleanPlan);
        this.cacheManager = NodeCacheManager.getInstance();
        this.connectionPool =
            DBConnectionPool.createDBConnectionPool(
                cleanPlan.getSourceDBConfig(),
                dbConfig
            );
        DBInstaller.install(dbConfig);

        context = ExecutionContext.createExecutorContext();
        context.setConnectionPool(this.connectionPool);
        context.setRule(cleanPlan.getRule());
        assembleFlow();
    }
    //</editor-fold>

    //<editor-fold desc="Incremental methods">
    public void incrementalAppend(String tableName, HashSet<Integer> newTuples) {
        context.addNewTuples(tableName, newTuples);
    }
    //</editor-fold>

    //<editor-fold desc="Public methods">

    /**
     * Returns <code>True</code> when the clean executor is running.
     * @return <code>True</code> when the clean executor is running.
     */
    public synchronized boolean isRunning() {
        return detectFlow.isRunning() ||
               queryFlow.isRunning() ||
               repairFlow.isRunning();
    }

    /**
     * Shutdown the CleanExecutor.
     */
    public void shutdown() {
        if (queryFlow != null) {
            if (queryFlow.isRunning()) {
                queryFlow.forceStop();
            }
        }

        queryFlow = null;

        if (detectFlow != null) {
            if (detectFlow.isRunning()) {
                detectFlow.forceStop();
            }
        }
        detectFlow = null;

        if (repairFlow != null) {
            if (repairFlow.isRunning()) {
                repairFlow.forceStop();
            }
        }

        if (connectionPool != null) {
            connectionPool.shutdown();
        }
        connectionPool = null;
    }

    /**
     * CleanExecutor finalizer.
     */
    @Override
    public void finalize() throws Throwable{
        shutdown();
        super.finalize();
    }

    /**
     * Gets the output from Detect.
     * @return output object from Detect.
     */
    public List<Violation> getDetectViolation() {
        String key = detectFlow.getCurrentOutputKey();
        return cacheManager.get(key);
    }

    /**
     * Gets the output from Repair.
     * @return output object from repair.
     */
    @SuppressWarnings("unchecked")
    public <T> T getRepairOutput() {
        String key = repairFlow.getCurrentOutputKey();
        return (T)cacheManager.get(key);
    }

    /**
     * Gets the current progress percentage of Detect.
     * @return current progress percentage of Detect.
     */
    public double getDetectProgress() {
        return detectFlow.getProgress();
        // return queryFlow.getProgress() * 0.5 + detectFlow.getProgress() * 0.5;
    }

    /**
     * Gets the detail progress information of Detection.
     * @return the detail progress information of Detection.
     */
    public List<ProgressReport> getDetailDetectProgress() {
        return detectFlow.getDetailProgress();
        // List<ProgressReport> queryProgress = queryFlow.getDetailProgress();
        // List<ProgressReport> detectProgress = detectFlow.getDetailProgress();
        // queryProgress.addAll(detectProgress);
        // return queryProgress;
    }

    /**
     * Gets the current percentage of Repair.
     * @return current percentage of Repair.
     */
    public double getRepairProgress() {
        return repairFlow.getProgress();
    }

    /**
     * Gets the detail progress information of Detection.
     * @return the detail progress information of Detection.
     */
    public List<ProgressReport> getDetailRepairProgress() {
        return repairFlow.getDetailProgress();
    }

    /**
     * Runs the violation detection.
     */
    public CleanExecutor detect() {
        Stopwatch sw = Stopwatch.createStarted();

        // queryFlow.reset();
        detectFlow.reset();

        // queryFlow.start();
        detectFlow.start();

        // queryFlow.waitUntilFinish();
        detectFlow.waitUntilFinish();

        // clear the new tuples after every run.
        context.clearNewTuples();

        PerfReport.appendMetric(
            PerfReport.Metric.DetectTime,
            sw.elapsed(TimeUnit.MILLISECONDS)
        );

        // TODO: remove it.
        System.gc();
        return this;
    }

    /**
     * Gets the CleanPlan.
     * @return the CleanPlan.
     */
    public CleanPlan getCleanPlan() {
        return cleanPlan;
    }

    /**
     * Runs the violation repair.
     */
    public CleanExecutor repair() {
        Stopwatch sw = Stopwatch.createStarted();
        repairFlow.reset();

        repairFlow.start();
        repairFlow.waitUntilFinish();

        context.clearNewTuples();

        PerfReport.appendMetric(
            PerfReport.Metric.RepairTime,
            sw.elapsed(TimeUnit.MILLISECONDS)
        );
        sw.stop();
        // TODO: remove it.
        System.gc();
        return this;
    }

    /**
     * Runs both the detection and repair.
     */
    public synchronized CleanExecutor run() {
        detect();
        repair();
        return this;
    }
    //</editor-fold>

    //<editor-fold desc="Private members">
    /**
     * Assemble the workflow on demand.
     */
    @SuppressWarnings("unchecked")
    private void assembleFlow() {
        try {
            // assemble the query flow.
            /*
            queryFlow = new Flow("query");
            queryFlow
                .setInputKey(cacheManager.getKeyForNothing())
                .addNode(new SourceImport(context))
                .addNode(new ScopeOperator(context))
                .addNode(new Iterator(context));

            // assemble the detect flow
            detectFlow = new Flow("detect");
            detectFlow
                .setInputKey(cacheManager.getKeyForNothing())
                .addNode(new ViolationDetector(context), 6)
                // .addNode(new ViolationExport(context))
                .addNode(new ViolationExportToCSV(context))
                .addNode(new ViolationCSVExport(context));
            */

            detectFlow = new Flow("detect");
            detectFlow
                .setInputKey(cacheManager.getKeyForNothing())
                .addNode(new SourceImport(context))
                .addNode(new ScopeOperator(context))
                .addNode(new DirectIterator(context))
                .addNode(new ViolationExportToCSV(context))
                .addNode(new ViolationCSVExport(context));

            // assemble the repair flow
            repairFlow = new Flow("repair");
            repairFlow.setInputKey(cacheManager.getKeyForNothing())
                .addNode(new ViolationImport(context))
                .addNode(new ViolationRepair(context), 6)
                .addNode(new FixExport(context));

        } catch (Exception ex) {
            tracer.error("Exception happens during assembling the pipeline ", ex);
        }
    }
    //</editor-fold>
}
