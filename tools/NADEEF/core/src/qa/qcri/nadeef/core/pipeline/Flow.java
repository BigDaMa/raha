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
import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.ProgressReport;
import qa.qcri.nadeef.tools.Logger;

import java.util.List;

/**
 * Flow state.
 */
enum FlowState {
    Ready,
    Running,
    Stopped,
    StoppedWithException
}

/**
 * Flow contains a series of Nodes which can be connected and executed in sequence.
 * Currently the design only allows for one line of connected nodes.
 *
 */
public class Flow {
    //<editor-fold desc="Private fields">
    private static Logger tracer = Logger.getLogger(Flow.class);

    private List<Node> nodeList;
    private List<Integer> weights;
    private List<String> keyList;
    private int currentFlowPosition;
    private FlowState state;
    private String name;
    private String inputKey;
    private Thread thread;
    private boolean forceStop = false;
    //</editor-fold>

    //<editor-fold desc="Constructor">
    /**
     * Constructor.
     * @param name the name of the flow.
     */
    public Flow(String name) {
        this.name = name;
        nodeList = Lists.newArrayList();
        keyList = Lists.newArrayList();
        weights = Lists.newArrayList();
        currentFlowPosition = 0;
        state = FlowState.Ready;
    }
    //</editor-fold>

    //<editor-fold desc="Public methods">

    /**
     * Resets the flow.
     */
    public void reset() {
        if (thread != null && thread.isAlive()) {
            throw new RuntimeException("Flow cannot be reset during running.");
        }
        state = FlowState.Ready;
        currentFlowPosition = 0;
        for (int i = 0; i < nodeList.size(); i ++) {
            nodeList.get(i).reset();
        }
    }

    /**
     * Sets the input key of the flow.
     * @param inputKey input key.
     */
    public Flow setInputKey(String inputKey) {
        this.inputKey = inputKey;
        return this;
    }

    /**
     * Adds a new node in the flow.
     * @param node node.
     * @param index index.
     */
    public Flow addNode(Node node, int index) {
        return addNode(node, index, 1);
    }

    /**
     * Adds an operator in the flow.
     * @param operator operator.
     * @param weight percentage weight of the flow.
     * @return Flow itself.
     */
    public Flow addNode(Operator operator, int weight) {
        return addNode(
            new Node(operator, operator.getClass().getSimpleName()),
            nodeList.size(),
            weight
        );
    }

    /**
     * Adds an operator in the flow.
     * @param operator operator.
     * @return Flow itself.
     */
    public Flow addNode(Operator operator) {
        return addNode(
            new Node(operator, operator.getClass().getSimpleName()),
            nodeList.size(),
            1
        );
    }

    /**
     * Adds an node in the flow.
     * @param node flow node.
     * @param index index.
     * @param weight percentage weight.
     * @return Flow itself.
     */
    public Flow addNode(Node node, int index, int weight) {
        if (thread != null && thread.isAlive()) {
            throw new RuntimeException("Flow cannot be modified during running.");
        }

        Preconditions.checkArgument(
            weight <= 100 && weight > 0,
            "Weight needs to be between [1, 100]."
        );
        nodeList.add(index, node);
        weights.add(weight);
        return this;
    }

    /**
     * Starts the flow.
     */
    public void start() {
        if (thread != null && thread.isAlive()) {
            tracer.info("Flow " + name + " is already started.");
        }

        state = FlowState.Running;
        thread = new Thread() {
            public void run() {
                try {
                    String inputKey_ = inputKey;
                    for (int i = currentFlowPosition; i < nodeList.size(); i ++) {
                        if (i != 0) {
                            inputKey_ = keyList.get(i - 1);
                        }

                        if (forceStop) {
                            break;
                        }

                        Node node = nodeList.get(i);
                        if (node.canExecute(inputKey_)) {
                            String outputKey = node.execute(inputKey_);
                            keyList.add(currentFlowPosition, outputKey);
                            currentFlowPosition ++;
                        } else {
                            state = FlowState.Stopped;
                            throw
                                new IllegalStateException(
                                    "Flow stops at node: " +
                                    nodeList.get(currentFlowPosition).getName()
                                );
                        }
                    }
                    state = FlowState.Stopped;
                } catch (Exception ex) {
                    Node curNode = nodeList.get(currentFlowPosition);
                    tracer.error("Flow stops at node " + curNode.getName(), ex);
                    state = FlowState.StoppedWithException;
                } finally {
                    for (int i = 0; i < nodeList.size(); i ++) {
                        nodeList.get(i).interrupt();
                    }
                }
            }
        };
        thread.start();
    }

    /**
     * Wait until the flow is finished. It is a blocking operation.
     */
    public void waitUntilFinish() {
        try {
            if (thread != null) {
                thread.join();
            }
        } catch (InterruptedException ex) {
            tracer.error("Flow " + name + " is interrupted.", ex);
        }
    }

    /**
     * Forces the Flow to be closed.
     */
    public synchronized void forceStop() {
        forceStop = true;
    }

    /**
     * Gets the last output key.
     * @return output key.
     */
    public String getCurrentOutputKey() {
        if (currentFlowPosition == 0) {
            return null;
        }
        return keyList.get(currentFlowPosition - 1);
    }

    /**
     * Returns <code>True</code> when the flow is still running.
     * @return <code>True</code> when the flow is still running.
     */
    public boolean isRunning() {
        return state == FlowState.Running;
    }

    /**
     * Gets the current progress percentage.
     */
    public double getProgress() {
        int percentage = 0;
        double weightSum = 0;
        int weight;
        for (int i = 0; i < weights.size(); i ++) {
            weight = weights.get(i);
            weightSum += weight;
            Node node = nodeList.get(i);
            percentage += node.getProgress() * weight;
        }
        return (double)percentage / weightSum;
    }

    /**
     * Gets the detail progress information for each operator.
     * @return detail progress information for each operator.
     */
    public List<ProgressReport> getDetailProgress() {
        List<ProgressReport> list = Lists.newArrayList();
        for (Node node : nodeList) {
            list.add(node.getDetailProgress());
        }
        return list;
    }

    //</editor-fold>
}
