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
import com.google.common.collect.Maps;

import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentMap;

/**
 * Node Cache manager manages the input/output of the node execution.
 * It is basically a pair-value container.
 */
public class NodeCacheManager {
    private static ConcurrentMap<String, Object> cachePool;
    private static ConcurrentMap<String, Integer> refPool;
    private static String absentKey;
    private static NodeCacheManager instance;
    //<editor-fold desc="Singleton">
    static {
        cachePool = Maps.newConcurrentMap();
        refPool = Maps.newConcurrentMap();
        instance = new NodeCacheManager();
        absentKey = instance.put(Optional.absent(), Integer.MAX_VALUE);
    }

    /**
     * Singleton instance.
     */
    public static NodeCacheManager getInstance() {
        return instance;
    }
    //</editor-fold>

    /**
     * Returns a key which refers to an absent value. It is used to feed
     * operator which needs no input from upper stream (e.g. start operator).
     */
    public String getKeyForNothing() {
        return absentKey;
    }

    /**
     * Add key-value pair in the container.
     * @param key value key.
     * @param value value.
     */
    public void put(String key, Object value) {
        if (cachePool.containsKey(key)) {
            throw new IllegalStateException("Invalid key, key already existed in the cache.");
        }

        cachePool.put(key, value);
        refPool.put(key, 1);
    }

    /**
     * Add a value with generated unique key.
     * @param value new object.
     * @return generated key.
     */
    public String put(Object value) {
        UUID uuId = UUID.randomUUID();
        String key = uuId.toString();
        put(key, value);
        return key;
    }

    /**
     * Add a value with generated unique key.
     * @param value new object.
     * @param lifeCount life count.
     * @return generated key.
     */
    public String put(Object value, int lifeCount) {
        UUID uuId = UUID.randomUUID();
        String key = uuId.toString();
        put(key, value, lifeCount);
        return key;
    }

    /**
     * Add key-value pair in the container.
     * @param key value key.
     * @param value value.
     * @param lifeCount life time of the value.
     */
    public void put(String key, Object value, int lifeCount) {
        if (cachePool.containsKey(key)) {
            throw new IllegalStateException("Invalid key, key already existed in the cache.");
        }

        cachePool.put(key, value);
        refPool.put(key, lifeCount);
    }

    /**
     * Get the value from key, without reducing the life count.
     * @param key
     * @return value.
     */
    public Object tease(String key) {
        if (!cachePool.containsKey(key)) {
            throw new IllegalStateException("Invalid key, key doesn't exist in the cache.");
        }

        return cachePool.get(key);
    }

    /**
     * Get the value from the key. Clean the object from the cache once the life
     * cycle is done.
     * @param key value key.
     * @return value.
     */
    @SuppressWarnings("unchecked")
    public <T> T get(String key) {
        Object result = tease(key);

        // clean the cache once the life is finished, to prevent memory leaking.
        int refCount = refPool.get(key) - 1;
        if (refCount == 0) {
            cachePool.remove(key);
            refPool.remove(key);
        } else {
            refPool.put(key, refCount);
        }

        return (T)result;
    }

    /**
     * Gets the size of the cache pool.
     * @return size of the cache pool.
     */
    public int getSize() {
        return cachePool.size();
    }

    /**
     * Clear all the resources in the cache.
     */
    public void clear() {
        Set<String> keys = cachePool.keySet();
        for (String key : keys) {
            if (!key.equalsIgnoreCase(absentKey)) {
                cachePool.remove(key);
                refPool.remove(key);
            }
        }
    }
}
