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

import qa.qcri.nadeef.core.datamodel.ProgressReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.UUID;

/**
 * A component runs in the pipeline.
 */
public class Node {

    //<editor-fold desc="Private fields">
    private static Logger tracer = Logger.getLogger(Node.class);
    private UUID uid;
    private String name;
    private Operator operator;

    private synchronized String generateKey() {
        return System.currentTimeMillis() + "_name_" + uid.toString();
    }

    //</editor-fold>

    //<editor-fold desc="Public methods">

    /**
     * Constructor.
     * @param operator Operator
     */
    public Node(Operator operator, String name) {
        // currently a node contains only one operator.
        this.operator = operator;
        uid = UUID.randomUUID();
        this.name = name;
    }

    public boolean canExecute(String key) {
        NodeCacheManager nodeCache = NodeCacheManager.getInstance();
        return operator.canExecute(nodeCache.tease(key));
    }

    @SuppressWarnings("unchecked")
    public String execute(String key) {
        Object result;
        if (operator == null) {
            throw new NullPointerException("Operator is null.");
        }

        NodeCacheManager nodeCache = NodeCacheManager.getInstance();
        Object input = nodeCache.get(key);
        // TODO: adds exception handling on node.
        if (operator.canExecute(input)) {
            try {
                result = operator.execute(input);
                operator.setPercentage(1.0f);
                String newKey = generateKey();
                nodeCache.put(newKey, result, 1);
                return newKey;
            } catch (Exception ex) {
                tracer.error("Node has an exception during execution.", ex);
            }
        }

        return null;
    }

    /**
     * Gets the current progress percentage.
     * @return progress percentage.
     */
    public double getProgress() {
        return operator.getPercentage();
    }

    /**
     * Gets detail progress information of the operator.
     * @return detail progress information of the operator.
     */
    public ProgressReport getDetailProgress() {
        double percentage = operator.getPercentage();
        return new ProgressReport(name, percentage);
    }

    /**
     * Gets the node name.
     * @return Node name
     */
    public String getName() {
        return name;
    }

    /**
     * Interrupt is called in situation when the operator needs to shutdown during running.
     */
    void interrupt() {
        operator.interrupt();
    }

    /**
     * Resets the operator.
     */
    void reset() {
        operator.reset();
    }

    //</editor-fold>
}
