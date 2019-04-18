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

import qa.qcri.nadeef.core.datamodel.Fix;

import java.util.Collection;

/**
 * FixDecisionMaker provides algorithm which selects the right candidate fix based on a
 * collection of @see Fix.
 *
 */
public abstract class FixDecisionMaker extends Operator<Collection<Fix>, Collection<Fix>> {
    /**
     * Constructor.
     */
    public FixDecisionMaker(ExecutionContext context) {
        super(context);
    }

    /**
     * Decides which fixes are right given a collection of candidate fixes.
     *
     * @param fixes candidate fixes.
     * @return a collection of right {@see Fix}.
     */
    public abstract Collection<Fix> decide(Collection<Fix> fixes);

    /**
     * {@inheritDoc}
     */
    @Override
    protected final Collection<Fix> execute(Collection<Fix> fixes) throws Exception {
        return decide(fixes);
    }
}
