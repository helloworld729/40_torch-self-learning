a=\
    {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "id": "http://www.iptc.org/std/ninjs/ninjs-schema_1.2.json#",
    "type": "object",
    "title": "IPTC ninjs - News in JSON - version 1.2 (approved, 2019-10-16)",
    "description": "A news item as JSON object -- copyright 2019 IPTC - International Press Telecommunications Council - www.iptc.org - This document is published under the Creative Commons Attribution 4.0 license, see  http://creativecommons.org/licenses/by/4.0/  $$comment: as of 2019-10-16 ",
    "additionalProperties": "false",
    "required": ["uri"],
    "patternProperties": {
        "^description_[a-zA-Z0-9_]+": {
            "title": "Description",
            "description": "A free-form textual description of the content of the item. (The string appended to description_ in the property name should reflect the format of the text). nar:description",
            "type": "string"
        },
        "^body_[a-zA-Z0-9_]+": {
            "title": "Body",
            "description": "The textual content of the news object. (The string appended to body_ in the property name should reflect the format of the text). nar:inlineData or nar:inlineXML",
            "type": "string"
        }
    },
    "properties": {
        "uri": {
            "title": "Uniform Resource Identifier",
            "description": "The identifier for this news object. nar:newsItem@guid",
            "type": "string",
            "format": "uri"
        },
        "type": {
            "title": "Type",
            "description": "The generic news type of this news object. (Value 'component' added in version 1.2 as issue #21.). nar:itemClass",
            "type": "string",
            "enum": [
                "text",
                "audio",
                "video",
                "picture",
                "graphic",
                "composite",
                "component"
            ]
        },
        "mimetype": {
            "title": "MIME type",
            "description": "A MIME type which applies to this news object. nar:contentType",
            "type": "string"
        },
        "representationtype": {
            "title": "Representation type",
            "description": "Indicates how complete this representation of a news item is. No mapping to nar. Specific for ninjs.",
            "type": "string",
            "enum": [
                "complete",
                "incomplete"
            ]
        },
        "profile": {
            "title": "Profile",
            "description": "An identifier for the kind of content of this news object. This can be any string but we suggest something identifying the content such as 'text-only' or 'text-photo'. (Investigate if this align with NewsML nar:profile or not.)",
            "type": "string"
        },
        "version": {
            "title": "Version",
            "description": "The version of the news object which is identified by the uri property. nar:newsItem@version",
            "type": "string"
        },
        "firstcreated": {
            "title": "First created",
            "description": "Indicates when the first version of the item was created. (Added in version 1.2 from issue #5). nar:firstCreated",
            "type": "string",
            "format": "date-time"
        },
        "versioncreated": {
            "title": "Version created",
            "description": "The date and time when this version of the news object was created. nar:versionCreated",
            "type": "string",
            "format": "date-time"
        },
        "embargoed": {
            "title": "Embargoed",
            "description": "The date and time before which all versions of the news object are embargoed. If absent, this object is not embargoed. nar:embargoed",
            "type": "string",
            "format": "date-time"
        },
        "pubstatus": {
            "title": "Publication status",
            "description": "The publishing status of the news object, its value is *usable* by default. nar:pubStatus",
            "type": "string",
            "enum": [
                "usable",
                "withheld",
                "canceled"
            ]
        },
        "urgency": {
            "title": "Urgency",
            "description": "The editorial urgency of the content from 1 to 9. 1 represents the highest urgency, 9 the lowest. nar:urgency",
            "type": "number"
        },
        "copyrightholder": {
            "title": "Copyright holder",
            "description": "The person or organisation claiming the intellectual property for the content. nar:copyrightHolder",
            "type": "string"
        },
        "copyrightnotice": {
            "title": "Copyright notice",
            "description": "Any necessary copyright notice for claiming the intellectual property for the content. nar:copyrightNotice",
            "type": "string"
        },
        "usageterms": {
            "title": "Usage terms",
            "description": "A natural-language statement about the usage terms pertaining to the content. nar:usageTerms",
            "type": "string"
        },
        "ednote": {
            "title": "Editorial note",
            "description": "A note that is intended to be read by internal staff at the receiving organisation, but not published to the end-user. (Added in version 1.2 from issue #6.) . ednote: nar:edNote",
            "type": "string"
        },
        "language": {
            "title": "Language",
            "description": "The human language used by the content. The value should follow IETF BCP47. nar:language",
            "type": "string"
        },
        "person": {
            "title": "Person",
            "description": "An individual human being. nar:subject",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of a person",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the person",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the person",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the person in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "organisation": {
            "title": "Organisation",
            "description": "An administrative and functional structure which may act as as a business, as a political party or not-for-profit party. nar:subject",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the organisation",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the organisation",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the organisation",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the organisation in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    },
                    "symbols": {
                        "title": "Symbols",
                        "description": "Symbols used for a finanical instrument linked to the organisation at a specific market place",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": "false",
                            "properties": {
                                "ticker": {
                                    "title": "Ticker",
                                    "description": "Ticker symbol used for the financial instrument",
                                    "type": "string"
                                },
                                "exchange": {
                                    "title": "Exchange",
                                    "description": "Identifier for the marketplace which uses the ticker symbols of the ticker property",
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            }
        },
        "place": {
            "title": "Place",
            "description": "A named location. nar:subject",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "patternProperties": {
                    "^geometry_[a-zA-Z0-9_]+": {
                        "description": "An object holding geo data of this place. Could be of any relevant geo data JSON object definition.",
                        "type": "object"
                    }
                },
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the place",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the place",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the place",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the place in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "subject": {
            "title": "Subject",
            "description": "A concept with a relationship to the content. nar:subject",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the subject",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the subject",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the subject",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the subject in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "event": {
            "title": "Event",
            "description": "Something which happens in a planned or unplanned manner. nar:?",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the event",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the event",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the event",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the event in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "object": {
            "title": "Object",
            "description": "Something material, excluding persons. nar:subject",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the object",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the object",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Scheme",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the object",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the object in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "infosource": {
            "title": "Info source",
            "description": "A party (person or organisation) which originated, modified, enhanced, distributed, aggregated or supplied the content or provided some information used to create or enhance the content. (Added in version 1.2 according to issue #15.) .    infosource:  nar:infoSource",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": "false",
                "properties": {
                    "name": {
                        "title": "Name",
                        "description": "The name of the infosource",
                        "type": "string"
                    },
                    "rel": {
                        "title": "Relationship",
                        "description": "The relationship of the content of the news object to the infosource",
                        "type": "string"
                    },
                    "scheme": {
                        "title": "Schema",
                        "description": "The identifier of a scheme (= controlled vocabulary) which includes a code for the infosource",
                        "type": "string",
                        "format": "uri"
                    },
                    "code": {
                        "title": "Code",
                        "description": "The code for the infosource in a scheme (= controlled vocabulary) which is identified by the scheme property",
                        "type": "string"
                    }
                }
            }
        },
        "title": {
            "title": "Title",
            "description": "A short natural-language name for the item. (Added in version 1.2 according to issue #9). nar:itemMeta/title",
            "type": "string"
        },
        "byline": {
            "title": "Byline",
            "description": "The name(s) of the creator(s) of the content. nar:by",
            "type": "string"
        },
        "headline": {
            "title": "Headline",
            "description": "A brief and snappy introduction to the content, designed to catch the reader's attention. nar:headline",
            "type": "string"
        },
        "slugline": {
            "title": "Slugline",
            "description": "A human-readable identifier for the item. (Added in version 1.2 from issue #4.). nar:slugline",
            "type": "string"
        },
        "located": {
            "title": "Located",
            "description": "The name of the location from which the content originates. nar:located",
            "type": "string"
        },
        "charcount": {
            "title": "Character count",
            "description": "The total character count in the article excluding figure captions. (Added in version 1.2 according to issue #27.). nar:charcount",
            "type": "number"
        },
        "wordcount": {
            "title": "Word count",
            "description": "The total number of words in the article excluding figure captions. (Added in version 1.2 according to issue #27.). nar:wordcount",
            "type": "number"
        },
        "renditions": {
            "title": "Renditions",
            "description": "Wrapper for different renditions of the news object. nar:remoteContent",
            "type": "object",
            "additionalProperties": "false",
            "patternProperties": {
                "^[a-zA-Z0-9]+": {
                    "description": "A specific rendition of the content of the news object. (Description changed in version 1.2 according to issue #17.)",
                    "type": "object",
                    "additionalProperties": "false",
                    "properties": {
                        "href": {
                            "title": "href", 
                            "description": "The URL for accessing the rendition as a resource. nar:remoteContent@ref",
                            "type": "string",
                            "format": "uri"
                        },
                        "mimetype": {
                            "title": "mimetype", 
                            "description": "A MIME type which applies to the rendition. nar:remoteContent@contenttype",
                            "type": "string"
                        },
                        "title": {
                            "title": "Title", 
                            "description": "A title for the link to the rendition resource",
                            "type": "string"
                        },
                        "height": {
                            "title": "Height",
                            "description": "For still and moving images: the height of the display area measured in pixels. nar:remoteContent@height",
                            "type": "number"
                        },
                        "width": {
                            "title": "Width",
                            "description": "For still and moving images: the width of the display area measured in pixels. nar:remoteContent@width",
                            "type": "number"
                        },
                        "sizeinbytes": {
                            "title": "Size in bytes", 
                            "description": "The size of the rendition resource in bytes",
                            "type": "number"
                        },
                        "duration": {
                            "title": "Duration",
                            "description": "The total time duration of the content in seconds. (Added in version 1.2. Issue #18). nar:remoteContent@duration",
                            "type": "number"
                        },
                        "format": {
                            "title": "Format",
                            "description": "Binary format name. (Added in version 1.2. Issue #18). nar:remoteContent@format",
                            "type": "string"
                        }
                    }
                }
            }
        },
        "associations": {
            "description": "Content of news objects which are associated with this news object. nar:link",
            "type": "object",
            "additionalProperties": "false",
            "patternProperties": {
                "^[a-zA-Z0-9]+": {"$ref": "http://www.iptc.org/std/ninjs/ninjs-schema_1.2.json#"}
            }
        }
    }
}