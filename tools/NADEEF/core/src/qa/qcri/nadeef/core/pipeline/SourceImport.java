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
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.SQLTable;
import qa.qcri.nadeef.core.datamodel.Table;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * SourceImport creates the input tables as the input of the pipeline.
 */
public class SourceImport extends Operator<Optional, Collection<Table>> {
    /**
     * Constructor.
     */
    public SourceImport(ExecutionContext context) {
        super(context);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Table> execute(Optional emptyInput) {
        ExecutionContext context = getCurrentContext();
        Rule rule = context.getRule();

        @SuppressWarnings("unchecked")
        List<String> tableNames = rule.getTableNames();
        List<Table> collections = new ArrayList<>();
        if (tableNames.size() == 2) {
            collections.add(new SQLTable(tableNames.get(0), context.getConnectionPool()));
            collections.add(new SQLTable(tableNames.get(1), context.getConnectionPool()));
        } else {
            collections.add(new SQLTable(tableNames.get(0), context.getConnectionPool()));
        }

        return collections;
    }
}
