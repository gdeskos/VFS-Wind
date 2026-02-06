/*****************************************************************
* Copyright (C) by Regents of the University of Minnesota.       *
*                                                                *
* This Software is released under GNU General Public License 2.0 *
* http://www.gnu.org/licenses/gpl-2.0.html                       *
*                                                                *
* Modernization of the code by G Deskos,                         *
* Parametrica Research & Analytics                               *
*                                                                *
******************************************************************/

#ifndef _XML_INPUT_H_
#define _XML_INPUT_H_

#ifdef ENABLE_XML_INPUT

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parse an XML control file and set simulation parameters.
 *
 * @param filename Path to the XML control file
 * @return 0 on success, non-zero on failure
 */
int ParseXMLControlFile(const char* filename);

/**
 * Check if a file exists.
 *
 * @param filename Path to the file
 * @return 1 if file exists, 0 otherwise
 */
int xml_file_exists(const char* filename);

#ifdef __cplusplus
}
#endif

#endif /* ENABLE_XML_INPUT */

#endif /* _XML_INPUT_H_ */
