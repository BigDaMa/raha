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

package qa.qcri.nadeef.core.datamodel;

import com.google.common.collect.Lists;
import qa.qcri.nadeef.tools.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Streaming output (Bounded Queued Buffer).
 */
public class IteratorBlockingQueue implements IteratorResultHandler {
    private static final long TIMEOUT;
    private static final int BUFFER_BOUNDARY;
    private static final int MAX_QUEUE_BOUNDARY;
    private static Logger tracer;
    private static LinkedBlockingQueue<List<Object>> queue;

    static {
        TIMEOUT = 1024;
        BUFFER_BOUNDARY = 10240;
        MAX_QUEUE_BOUNDARY = 1024;
        queue = new LinkedBlockingQueue<>(MAX_QUEUE_BOUNDARY);
        tracer = Logger.getLogger(IteratorBlockingQueue.class);
    }

    private List<Object> buffer;

    /**
     * Constructor.
     */
    public IteratorBlockingQueue() {
        this.buffer = Lists.newArrayList();
    }

    /**
     * Gets a buffer of objects from the queue.
     * @return a list of objects from the queue.
     */
    public List<Object> poll() {
        List<Object> item = null;
        try {
            while ((item = queue.poll(TIMEOUT, TimeUnit.MILLISECONDS)) == null);
        } catch (InterruptedException ex) {
            tracer.error("Exception during polling the queue.", ex);
        }
        return item;
    }

    /**
     * Marks the end of the iteration output.
     */
    public static void markEnd() {
        try {
            List<Object> end = new ArrayList<>(0);
            while (!queue.offer(end, TIMEOUT, TimeUnit.MILLISECONDS));
        } catch (InterruptedException ex) {
            tracer.error("Exception during marking the end of the queue.", ex);
        }
    }

    /**
     * Puts the item in the buffer.
     * @param item item.
     */
    @Override public <T> void handle(T item) {
        if (buffer.size() == BUFFER_BOUNDARY) {
            try {
                while (!queue.offer(new ArrayList<>(buffer), TIMEOUT, TimeUnit.MILLISECONDS));
            } catch (InterruptedException e) {
                tracer.error("offer interrupted", e);
            }
            buffer = Lists.newArrayList();
        }

        buffer.add(item);
    }

    /**
     * Flush the remaining buffer.
     */
    public void flush() {
        try {
            if (buffer.size() != 0) {
                while (!queue.offer(new ArrayList<>(buffer), TIMEOUT, TimeUnit.MILLISECONDS));
            }
            buffer = null;
        } catch (InterruptedException e) {
            tracer.error("flush interrupted", e);
        }
    }

    /**
     * Clear the buffer queue.
     */
    public static void clear() {
        queue.clear();
    }
}
