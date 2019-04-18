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
import com.google.common.collect.Maps;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;

import java.util.HashSet;
import java.util.concurrent.ConcurrentMap;

/**
 * Execution context class which contains the shared variables during detection / repairing.
 */
public class ExecutionContext {
    private ConcurrentMap<String, HashSet<Integer>> newTuples;
    private DBConnectionPool connectionPool;
    private Rule rule;

    private ExecutionContext() {
        newTuples = Maps.newConcurrentMap();
    }

    static ExecutionContext createExecutorContext() {
        return new ExecutionContext();
    }

    //<editor-fold desc="Incremental new tuple methods">
    void clearNewTuples() {
        newTuples.clear();
    }

    public ConcurrentMap<String, HashSet<Integer>> getNewTuples() {
        return newTuples;
    }

    void addNewTuples(String tableName, HashSet<Integer> newTupleIds) {
        newTuples.put(tableName, newTupleIds);
    }
    //</editor-fold>

    public Rule getRule() {
        if (rule == null) {
            throw new RuntimeException("Rule in the context is not initialized.");
        }
        return rule;
    }

    void setRule(Rule rule) {
        this.rule = Preconditions.checkNotNull(rule);
    }

    DBConnectionPool getConnectionPool() {
        if (connectionPool== null) {
            throw new RuntimeException("Connection pool in the context is not initialized.");
        }
        return connectionPool;
    }

    void setNewTuples(ConcurrentMap<String, HashSet<Integer>> newTuples) {
        this.newTuples = newTuples;
    }

    void setConnectionPool(DBConnectionPool connectionPool) {
        this.connectionPool = Preconditions.checkNotNull(connectionPool);
    }
}
