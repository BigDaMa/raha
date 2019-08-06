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
import com.google.common.base.Stopwatch;
import qa.qcri.nadeef.core.datamodel.CleanPlan;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

/**
 * Updater flow executor.
 */
public class UpdateExecutor {
    private Flow updateFlow;
    private NodeCacheManager cacheManager;
    private Logger tracer;
    private DBConnectionPool connectionPool;
    private ExecutionContext context;

    public UpdateExecutor(CleanPlan cleanPlan) {
        this(cleanPlan, NadeefConfiguration.getDbConfig());
    }

    public UpdateExecutor(CleanPlan cleanPlan, DBConfig nadeefConfig) {
        cacheManager = NodeCacheManager.getInstance();
        tracer = Logger.getLogger(UpdateExecutor.class);
        this.connectionPool =
            DBConnectionPool.createDBConnectionPool(
                cleanPlan.getSourceDBConfig(),
                nadeefConfig
            );

        context = ExecutionContext.createExecutorContext();
        context.setConnectionPool(connectionPool);
        context.setRule(cleanPlan.getRule());
        assembleFlow();
    }

    public void shutdown() {
        if (updateFlow != null) {
            if (updateFlow.isRunning()) {
                updateFlow.forceStop();
            }
        }

        updateFlow = null;

        if (connectionPool != null) {
            connectionPool.shutdown();
        }
        connectionPool = null;
    }

    @Override
    public void finalize() throws Throwable{
        shutdown();
        super.finalize();
    }

    /**
     * Gets the updated cell count.
     * @return updated cell count.
     */
    public int getUpdateCellCount() {
        String key = updateFlow.getCurrentOutputKey();
        Object output = cacheManager.get(key);
        return ((ArrayList) output).size();
    }

    public void run() {
        Stopwatch sw = Stopwatch.createStarted();
        updateFlow.reset();
        updateFlow.start();
        updateFlow.waitUntilFinish();

        PerfReport.appendMetric(PerfReport.Metric.EQTime, sw.elapsed(TimeUnit.MILLISECONDS));
        sw.stop();
    }

    @SuppressWarnings("unchecked")
    private void assembleFlow() {
        try {
            // assemble the updater flow
            updateFlow = new Flow("update");
            Optional<Class> eqClass = NadeefConfiguration.getDecisionMakerClass();
            // check whether user provides a customized DecisionMaker class, if so, replace it
            // with default EQ class.
            FixDecisionMaker fixDecisionMaker = null;
            if (eqClass.isPresent()) {
                Class customizedClass = eqClass.get();
                if (!FixDecisionMaker.class.isAssignableFrom(customizedClass)) {
                    throw
                        new IllegalArgumentException(
                            "FixDecisionMaker class is not a class inherit from FixDecisionMaker"
                        );
                }

                Constructor<?>[] constructors = customizedClass.getConstructors();
                fixDecisionMaker = (FixDecisionMaker)constructors[0].newInstance(context);
            } else {
                fixDecisionMaker = new EquivalentClass(context);
            }

            updateFlow.setInputKey(cacheManager.getKeyForNothing())
                .addNode(new FixImport(context))
                .addNode(fixDecisionMaker, 6)
                .addNode(new Updater(context));
                // .addNode(new IncrementalUpdate(NadeefConfiguration.getDbConfig()));
        } catch (Exception ex) {
            tracer.error("Exception happens during assembling the update flow.", ex);
        }
    }
}
