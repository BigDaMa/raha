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
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.PerfReport;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.util.Collection;
import java.util.concurrent.TimeUnit;

/**
 * Export violations into the target place.
 */
public class ViolationExport extends Operator<Collection<Violation>, Collection<Violation>> {

    /**
     * Constructor.
     */
    public ViolationExport(ExecutionContext context) {
        super(context);
    }

    /**
     * Export the violation into database.
     *
     * @param violations violations.
     * @return whether the exporting is successful or not.
     */
    @Override
    public Collection<Violation> execute(Collection<Violation> violations) throws Exception {
        Stopwatch stopwatch = Stopwatch.createStarted();
        Connection conn = null;
        PreparedStatement stat = null;
        DBConnectionPool connectionPool = getCurrentContext().getConnectionPool();
        int count = 0;
        try {
            synchronized (ViolationExport.class) {
                // TODO: this is not out-of-process safe.
                int vid = Violations.generateViolationId(connectionPool);

                conn = connectionPool.getNadeefConnection();
                stat = conn.prepareStatement("INSERT INTO VIOLATION VALUES (?, ?, ?, ?, ?, ?)");

                for (Violation violation : violations) {
                    count ++;
                    Collection<Cell> cells = violation.getCells();
                    for (Cell cell : cells) {
                        // skip the tuple id
                        if (cell.hasColumnName("tid")) {
                            continue;
                        }
                        stat.setInt(1, vid);
                        stat.setString(2, violation.getRuleId());
                        stat.setString(3, cell.getColumn().getTableName());
                        stat.setInt(4, cell.getTid());
                        stat.setString(5, cell.getColumn().getColumnName());
                        Object value = cell.getValue();
                        if (value == null) {
                            stat.setString(6, null);
                        } else {
                            stat.setString(6, value.toString());
                        }
                        stat.addBatch();
                    }

                    if (count % 4096 == 0) {
                        stat.executeBatch();
                    }
                    vid ++;
                }
                setPercentage(0.5f);
                stat.executeBatch();
                conn.commit();
            }

            PerfReport.appendMetric(
                PerfReport.Metric.ViolationExportTime,
                stopwatch.elapsed(TimeUnit.MILLISECONDS)
            );

            PerfReport.appendMetric(
                PerfReport.Metric.ViolationExport,
                count
            );
            stopwatch.stop();
        } finally {
            if (stat != null) {
                stat.close();
            }
            if (conn != null) {
                conn.close();
            }
        }
        return violations;
    }
}
