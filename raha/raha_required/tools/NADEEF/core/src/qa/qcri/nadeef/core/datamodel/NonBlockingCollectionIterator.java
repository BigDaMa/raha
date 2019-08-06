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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class NonBlockingCollectionIterator<T> implements Iterator<T> {
    private List<Collection<T>> collections;
    private Iterator<T> currentIterator;
    private int currentCollection;

    public NonBlockingCollectionIterator() {
        this.collections = new ArrayList<>();
        this.currentCollection = 0;
    }

    public synchronized void appendCollection(Collection<T> collection) {
        this.collections.add(collection);
    }

    @Override
    public boolean hasNext() {
        return (collections.size() > 0 && currentCollection < collections.size()) ||
            (currentIterator != null && currentIterator.hasNext());
    }

    @Override public T next() {
        if (!hasNext()) {
            throw new IllegalStateException("Iterator reaches the end.");
        }

        T result;
        if (currentIterator != null && currentIterator.hasNext()) {
            result = currentIterator.next();
        } else {
            currentIterator = collections.get(currentCollection ++).iterator();
            result = next();
        }
        return result;
    }
}
