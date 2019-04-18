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
import com.google.common.reflect.TypeToken;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

/**
 * Abstract class for an Operator.
 */
public abstract class Operator<TInput, TOutput> {
    private double percentage;
    private TypeToken typeToken;
    private ExecutionContext context;

    /**
     * Constructor.
     */
    public Operator(ExecutionContext context) {
        this.typeToken = new TypeToken<TInput>(getClass()){};
        this.context = context;
    }

    /**
     * Gets the generic input class information at runtime.
     * @return inputType class.
     */
    @SuppressWarnings("all")
    public Type getInputType() {
        Type genericSuperType = getClass().getGenericSuperclass();
        if (!(genericSuperType instanceof ParameterizedType)) {
            return typeToken.getRawType();
        }
        ParameterizedType parameterizedType = (ParameterizedType)genericSuperType;

        Type[] types = parameterizedType.getActualTypeArguments();
        // loop all the type signature to find the actual input type.
        for (Type type : types) {
            if (type instanceof ParameterizedType) {
                return ((ParameterizedType)type).getRawType();
            } else {
                return type;
            }
        }
        throw new IllegalStateException("Input type is not identified.");
    }

    protected ExecutionContext getCurrentContext() {
        return context;
    }

    /**
     * Execute the operator.
     * @param input input object.
     * @return output object.
     */
    protected abstract TOutput execute(TInput input) throws Exception;

    /**
     * Gets the current progress percentage of this operator [0 - 1].
     * @return percentage in integer.
     */
    double getPercentage() { return percentage; }

    /**
     * Reset is called before operator starts to function.
     */
    void reset() {
        this.percentage = 0.0f;
    }

    /**
     * Interrupt is called in situation when the operator needs to shutdown during running.
     */
    void interrupt() {}

    /**
     * Sets the percentage.
     */
    void setPercentage(double percentage) {
        Preconditions.checkArgument(percentage >= 0.0f && percentage <= 1.0f);
        this.percentage = percentage;
    }

    /**
     * Check whether the operator is executable.
     * @param input input object.
     * @return True when the operator can execute on the given input object.
     */
    boolean canExecute(Object input) {
        Type inputClass = getInputType();
        return ((Class)inputClass).isInstance(input);
    }
}