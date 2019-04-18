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
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Operator which executes the repair of a rule.
 */
public class ViolationRepair
    extends Operator<Collection<Violation>, Collection<Collection<Fix>>> {

    public ViolationRepair(ExecutionContext context) {
        super(context);
    }

    /**
     * Execute the operator.
     *
     * @param violations input object.
     * @return output object.
     */
    @Override
    @SuppressWarnings("unchecked")
    public Collection<Collection<Fix>> execute(Collection<Violation> violations)
        throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        Rule rule = getCurrentContext().getRule();
        List<Collection<Fix>> result = Lists.newArrayList();
        int count = 0;
        for (Violation violation : violations) {
            try {
                Collection<Fix> fix = (Collection<Fix>)rule.repair(violation);
                result.add(fix);
                count ++;
            } catch (Exception ex) {
                Logger tracer = Logger.getLogger(ViolationRepair.class);
                tracer.error("Exception in repair method.", ex);
            }
            setPercentage(count / violations.size());
        }

        long elapseTime;
        if (violations.size() != 0) {
            elapseTime = stopwatch.elapsed(TimeUnit.MILLISECONDS) / violations.size();
        } else {
            elapseTime = stopwatch.elapsed(TimeUnit.MILLISECONDS);
        }

        PerfReport.appendMetric(PerfReport.Metric.RepairCallTime, elapseTime);
        stopwatch.stop();
        return result;
    }
}
