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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.mysql.jdbc.log.Log;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentMap;

/**
 * Updater fixes the source data and exports it in the database.
 */
public class Updater extends Operator<Collection<Fix>, Collection<Fix>> {
    private static Logger tracer = Logger.getLogger(Updater.class);
    private ConcurrentMap<Cell, String> updateHistory;
    private ConcurrentMap<Cell, Boolean> unknownTag;

    /**
     * Constructor.
     */
    public Updater(ExecutionContext context) {
        super(context);
        updateHistory = Maps.newConcurrentMap();
        unknownTag = Maps.newConcurrentMap();
    }

    /**
     * Apply the fixes from EQ and modify the original source.
     *
     * @param fixes real fix collection.
     * @return real fix collection.
     */
    @Override
    public Collection<Fix> execute(Collection<Fix> fixes) throws Exception {
        int count = 0;
        Connection sourceConn = null;
        Connection nadeefConn = null;
        Statement sourceStat = null;
        PreparedStatement auditStat = null;
        String auditTableName = NadeefConfiguration.getAuditTableName();
        String rightValue;
        String oldValue;
        ExecutionContext context = getCurrentContext();
        DBConnectionPool connectionPool = context.getConnectionPool();
        List<Fix> realFixes = Lists.newArrayList();
        try {
            nadeefConn = connectionPool.getNadeefConnection();
            sourceConn = connectionPool.getSourceConnection();
            sourceStat = sourceConn.createStatement();
            auditStat =
                nadeefConn.prepareStatement(
                    "INSERT INTO " + auditTableName +
                    " VALUES (default, ?, ?, ?, ?, ?, ?, current_timestamp)");
            for (Fix fix : fixes) {
                Cell cell = fix.getLeft();
                Object oldValue_ = cell.getValue();
                if (oldValue_ == null) {
                    oldValue = null;
                } else {
                    oldValue = oldValue_.toString();
                }

                // this cell has already been changed to unknown
                if (unknownTag.containsKey(cell)) {
                    continue;
                }

                realFixes.add(fix);
                // check whether this cell has been changed before
                if (updateHistory.containsKey(cell)) {
                    String value = updateHistory.get(cell);
                    if (value.equals(fix.getRightValue())) {
                        continue;
                    }
                    // when a cell is set twice with different value,
                    // we set it to null for ambiguous value.
                    unknownTag.put(cell, true);
                    rightValue = "?";
                } else {
                    rightValue = fix.getRightValue();
                    updateHistory.put(cell, rightValue);
                }

                // check for numerical type.
                if (rightValue != null && !CommonTools.isNumericalString(rightValue)) {
                    rightValue = '\'' + rightValue + '\'';
                }

                if (oldValue != null && !CommonTools.isNumericalString(oldValue)) {
                    oldValue = '\'' + oldValue + '\'';
                }

                Column column = cell.getColumn();
                String tableName = column.getTableName();
                String updateSql =
                    "UPDATE " + tableName +
                    " SET " + column.getColumnName() + " = " + rightValue +
                    " WHERE tid = " + cell.getTid();
                tracer.fine(updateSql);
                sourceStat.addBatch(updateSql);
                auditStat.setInt(1, fix.getVid());
                auditStat.setInt(2, cell.getTid());
                auditStat.setString(3, column.getTableName());
                auditStat.setString(4, column.getColumnName());
                auditStat.setString(5, oldValue);
                auditStat.setString(6, rightValue);
                auditStat.addBatch();
                if (count % 4096 == 0) {
                    auditStat.executeBatch();
                    nadeefConn.commit();
                    sourceStat.executeBatch();
                    sourceConn.commit();
                }
                count ++;
                setPercentage(count / fixes.size());
            }
            sourceStat.executeBatch();
            auditStat.executeBatch();
            sourceConn.commit();
            nadeefConn.commit();
            PerfReport.appendMetric(PerfReport.Metric.UpdatedCellNumber, count);
        } finally {
            if (auditStat != null) {
                auditStat.close();
            }

            if (sourceStat != null) {
                sourceStat.close();
            }

            if (nadeefConn != null) {
                nadeefConn.close();
            }

            if (sourceConn != null) {
                sourceConn.close();
            }
        }
        return realFixes;
    }
}
