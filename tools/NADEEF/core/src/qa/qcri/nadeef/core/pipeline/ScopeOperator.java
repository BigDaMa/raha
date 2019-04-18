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
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Table;
import qa.qcri.nadeef.tools.PerfReport;

import java.util.Collection;
import java.util.concurrent.TimeUnit;

/**
 * Query engine operator, which generates optimized queries based on given hints.
 */
public class ScopeOperator extends Operator<Collection<Table>, Collection<Table>> {

    public ScopeOperator(ExecutionContext context) {
        super(context);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    @SuppressWarnings("unchecked")
    public Collection<Table> execute(Collection<Table> tables)
        throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        ExecutionContext context = getCurrentContext();
        Rule rule = context.getRule();

        // Scope
        // Here the horizontalScope needs to be called before vertical Scope since
        // it may needs the attributes which are going to be removed from verticals scope.
        Collection<Table> horizontalScopeResult = rule.horizontalScope(tables);
        setPercentage(0.5f);
        long time = stopwatch.elapsed(TimeUnit.MILLISECONDS);
        long currentTime;

        Collection<Table> verticalScopeResult =
            rule.verticalScope(horizontalScopeResult);

        currentTime = stopwatch.elapsed(TimeUnit.MILLISECONDS);

        PerfReport.appendMetric(PerfReport.Metric.HScopeTime, time);
        PerfReport.appendMetric(PerfReport.Metric.VScopeTime, currentTime - time);
        PerfReport.appendMetric(PerfReport.Metric.AfterScopeTuple, verticalScopeResult.size());

        Collection<Table> result = verticalScopeResult;

        // Block
        // Currently we don't support co-group, so once a rule is working with two tables we
        // ignore the block function.
        if (!rule.supportTwoTables()) {
            result = rule.block(verticalScopeResult);
            PerfReport.appendMetric(PerfReport.Metric.Blocks, result.size());
        }

        stopwatch.stop();
        return result;
    }
}
