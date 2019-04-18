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

package qa.qcri.nadeef.console;

import jline.console.ConsoleReader;

import java.io.IOException;
import java.io.PrintStream;

/**
 * ConsoleReader to PrintStream adapter.
 *
 * @author Si Yin <siyin@qf.org.qa>
 */
public class ConsoleReaderAdaptor extends PrintStream {
    private ConsoleReader consoleReader;
    public ConsoleReaderAdaptor(ConsoleReader consoleReader) {
        super(System.out);
        this.consoleReader = consoleReader;
    }

    @Override
    public void println() {
        try {
            consoleReader.println();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void println(String string) {
        try {
            consoleReader.println(string);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void print(String string) {
        try {
            consoleReader.print(string);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void flush() {
        try {
            consoleReader.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
